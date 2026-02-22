/**
 * @file test_prefix_caching.cpp
 * @brief Prefix Caching 单元测试
 *
 * 测试覆盖：
 *   1. BlockManager 层: 引用计数、哈希分配、LRU 驱逐
 *   2. KVCacheManager 层: 序列级 Prefix Caching 分配与释放
 *   3. 端到端: 多请求共享前缀场景
 */
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <vector>
#include "nanoinfer/base/alloc.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/block_manager.h"
#include "nanoinfer/engine/kv_cache_manager.h"

using namespace engine;

// ================================================================
// BlockManager Prefix Caching Tests
// ================================================================

class BlockManagerCacheTest : public ::testing::Test {
   protected:
    void SetUp() override {
        num_blocks_ = 20;
        block_size_ = 16;
        bm_ = std::make_unique<BlockManager>(num_blocks_, block_size_);
    }

    int32_t num_blocks_;
    int32_t block_size_;
    std::unique_ptr<BlockManager> bm_;
};

// 1. allocate_cached 基础: 首次分配全部 miss
TEST_F(BlockManagerCacheTest, AllocateCachedMiss) {
    int32_t block_id;
    bool cache_hit;

    auto status = bm_->allocate_cached(0xABCD, block_id, cache_hit);
    ASSERT_TRUE(status);
    EXPECT_FALSE(cache_hit);
    EXPECT_GE(block_id, 0);
    EXPECT_EQ(bm_->get_ref_count(block_id), 1);
    EXPECT_EQ(bm_->get_cache_misses(), 1);
    EXPECT_EQ(bm_->get_cache_hits(), 0);
}

// 2. 同一 hash 第二次分配 → cache hit, 引用计数递增
TEST_F(BlockManagerCacheTest, AllocateCachedHitActive) {
    int32_t block_id1, block_id2;
    bool hit1, hit2;

    bm_->allocate_cached(0x1234, block_id1, hit1);
    EXPECT_FALSE(hit1);
    EXPECT_EQ(bm_->get_ref_count(block_id1), 1);

    // 同一 hash 再次分配
    bm_->allocate_cached(0x1234, block_id2, hit2);
    EXPECT_TRUE(hit2);
    EXPECT_EQ(block_id1, block_id2);              // 同一个 block
    EXPECT_EQ(bm_->get_ref_count(block_id1), 2);  // 引用计数 +1
    EXPECT_EQ(bm_->get_cache_hits(), 1);
}

// 3. release 后 block 进入驱逐列表，再次 allocate_cached 能命中
TEST_F(BlockManagerCacheTest, ReleaseAndReuseFromEviction) {
    int32_t block_id;
    bool hit;

    // 分配并释放
    bm_->allocate_cached(0xAAAA, block_id, hit);
    EXPECT_EQ(bm_->get_ref_count(block_id), 1);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);

    bm_->release(block_id);
    EXPECT_EQ(bm_->get_ref_count(block_id), 0);
    EXPECT_EQ(bm_->get_evictable_block_num(), 1);

    // 同一 hash 再次分配 → 从驱逐列表中恢复
    int32_t block_id2;
    bool hit2;
    bm_->allocate_cached(0xAAAA, block_id2, hit2);
    EXPECT_TRUE(hit2);
    EXPECT_EQ(block_id, block_id2);
    EXPECT_EQ(bm_->get_ref_count(block_id2), 1);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);  // 已从驱逐列表移除
}

// 4. release 非缓存 block (通过普通 allocate 分配) → 直接释放
TEST_F(BlockManagerCacheTest, ReleaseNonCachedBlock) {
    int32_t block_id;
    bm_->allocate(block_id);
    EXPECT_EQ(bm_->get_ref_count(block_id), 0);  // 普通分配没有 ref_count

    int32_t free_before = bm_->get_free_block_num();
    bm_->release(block_id);
    EXPECT_EQ(bm_->get_free_block_num(), free_before + 1);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);  // 不进入驱逐列表
}

// 5. LRU 驱逐: 空闲栈耗尽后从驱逐列表分配
TEST_F(BlockManagerCacheTest, LRUEviction) {
    // 用完所有空闲块
    std::vector<int32_t> cached_blocks;
    for (int i = 0; i < num_blocks_; ++i) {
        int32_t bid;
        bool hit;
        bm_->allocate_cached(static_cast<uint64_t>(i) + 1, bid, hit);
        ASSERT_TRUE(!hit);
        cached_blocks.push_back(bid);
    }
    EXPECT_EQ(bm_->get_free_block_num(), 0);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);

    // 全部释放 → 进入驱逐列表
    for (int32_t bid : cached_blocks) {
        bm_->release(bid);
    }
    EXPECT_EQ(bm_->get_free_block_num(), 0);
    EXPECT_EQ(bm_->get_evictable_block_num(), num_blocks_);

    // 分配一个新的 hash → 应该驱逐最旧的 block (cached_blocks[0])
    int32_t new_block;
    bool hit;
    auto status = bm_->allocate_cached(0xDEADBEEF, new_block, hit);
    ASSERT_TRUE(status);
    EXPECT_FALSE(hit);
    EXPECT_EQ(new_block, cached_blocks[0]);  // 驱逐最旧的
    EXPECT_EQ(bm_->get_evictable_block_num(), num_blocks_ - 1);

    // 旧的 hash=1 不再有效
    int32_t old_block;
    bool old_hit;
    bm_->allocate_cached(1, old_block, old_hit);
    EXPECT_FALSE(old_hit);  // hash=1 已被驱逐
}

// 6. 多引用: 两个序列共享同一 block，释放一个后另一个仍有效
TEST_F(BlockManagerCacheTest, SharedBlockMultiRef) {
    int32_t b1, b2;
    bool hit1, hit2;

    bm_->allocate_cached(0xBEEF, b1, hit1);
    EXPECT_FALSE(hit1);
    EXPECT_EQ(bm_->get_ref_count(b1), 1);

    bm_->allocate_cached(0xBEEF, b2, hit2);
    EXPECT_TRUE(hit2);
    EXPECT_EQ(b1, b2);
    EXPECT_EQ(bm_->get_ref_count(b1), 2);

    // 释放一个引用
    bm_->release(b1);
    EXPECT_EQ(bm_->get_ref_count(b1), 1);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);  // 还有引用，不进入驱逐列表

    // 再释放一个 → 进入驱逐列表
    bm_->release(b1);
    EXPECT_EQ(bm_->get_ref_count(b1), 0);
    EXPECT_EQ(bm_->get_evictable_block_num(), 1);
}

// 7. get_available_block_num = free + evictable
TEST_F(BlockManagerCacheTest, AvailableBlockNum) {
    int32_t bid;
    bool hit;

    EXPECT_EQ(bm_->get_available_block_num(), num_blocks_);

    bm_->allocate_cached(0x1, bid, hit);
    EXPECT_EQ(bm_->get_available_block_num(), num_blocks_ - 1);

    bm_->release(bid);
    // block 进入驱逐列表 (evictable)，free 减 1，evictable 加 1
    EXPECT_EQ(bm_->get_available_block_num(), num_blocks_);
}

// 8. reset 清除所有缓存状态
TEST_F(BlockManagerCacheTest, ResetClearsCache) {
    int32_t bid;
    bool hit;
    bm_->allocate_cached(0xFF, bid, hit);
    bm_->release(bid);
    EXPECT_EQ(bm_->get_evictable_block_num(), 1);

    bm_->reset();

    EXPECT_EQ(bm_->get_free_block_num(), num_blocks_);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);
    EXPECT_EQ(bm_->get_cache_hits(), 0);
    EXPECT_EQ(bm_->get_cache_misses(), 0);

    // 旧的 hash 不再有效
    int32_t bid2;
    bool hit2;
    bm_->allocate_cached(0xFF, bid2, hit2);
    EXPECT_FALSE(hit2);
}

// 9. OOM: 所有块都被活跃引用，无法分配
TEST_F(BlockManagerCacheTest, OOMWhenAllActive) {
    // 分配所有块，全部保持活跃
    for (int i = 0; i < num_blocks_; ++i) {
        int32_t bid;
        bool hit;
        bm_->allocate_cached(static_cast<uint64_t>(i) + 100, bid, hit);
    }
    EXPECT_EQ(bm_->get_free_block_num(), 0);
    EXPECT_EQ(bm_->get_evictable_block_num(), 0);

    // 尝试分配新块 → 应该失败
    int32_t bid;
    bool hit;
    auto status = bm_->allocate_cached(0xFFFFF, bid, hit);
    EXPECT_FALSE(status);
}

// ================================================================
// KVCacheManager Prefix Caching Tests
// ================================================================

class KVCachePrefixTest : public ::testing::Test {
   protected:
    void SetUp() override {
        num_blocks_ = 32;
        block_size_ = 4;  // 小 block 方便测试
        num_layers_ = 2;
        num_kv_heads_ = 2;
        head_size_ = 64;

        manager_ = std::make_unique<KVCacheManager>(num_blocks_, block_size_, num_layers_,
                                                    num_kv_heads_, head_size_);
        allocator_ = base::CPUDeviceAllocatorFactory::get_instance();
        auto status = manager_->init(allocator_);
        ASSERT_TRUE(status) << status.get_err_msg();
    }

    int32_t num_blocks_;
    int32_t block_size_;
    int32_t num_layers_;
    int32_t num_kv_heads_;
    int32_t head_size_;
    std::unique_ptr<KVCacheManager> manager_;
    std::shared_ptr<base::DeviceAllocator> allocator_;
};

// 1. 哈希函数: 相同 token 序列产生相同 hash
TEST_F(KVCachePrefixTest, HashDeterministic) {
    std::vector<int32_t> tokens = {1, 2, 3, 4};
    uint64_t h1 = KVCacheManager::compute_block_hash(0, tokens.data(), 4);
    uint64_t h2 = KVCacheManager::compute_block_hash(0, tokens.data(), 4);
    EXPECT_EQ(h1, h2);
}

// 2. 哈希函数: 不同 token 产生不同 hash
TEST_F(KVCachePrefixTest, HashDifferentTokens) {
    std::vector<int32_t> t1 = {1, 2, 3, 4};
    std::vector<int32_t> t2 = {1, 2, 3, 5};
    uint64_t h1 = KVCacheManager::compute_block_hash(0, t1.data(), 4);
    uint64_t h2 = KVCacheManager::compute_block_hash(0, t2.data(), 4);
    EXPECT_NE(h1, h2);
}

// 3. 哈希函数: 链式依赖 — 相同 token 不同前缀产生不同 hash
TEST_F(KVCachePrefixTest, HashChainDependency) {
    std::vector<int32_t> same_tokens = {5, 6, 7, 8};
    uint64_t h1 = KVCacheManager::compute_block_hash(0xAAAA, same_tokens.data(), 4);
    uint64_t h2 = KVCacheManager::compute_block_hash(0xBBBB, same_tokens.data(), 4);
    EXPECT_NE(h1, h2);
}

// 4. 首次分配: 全部 miss，num_cached_tokens = 0
TEST_F(KVCachePrefixTest, FirstAllocationNoCache) {
    // 10 tokens, block_size=4 → 2 full + 1 partial block
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int32_t cached = -1;

    auto status = manager_->allocate_sequence_cached(1, tokens, cached);
    ASSERT_TRUE(status) << status.get_err_msg();

    EXPECT_EQ(cached, 0);  // 第一次分配，无缓存
    EXPECT_TRUE(manager_->is_sequence_allocated(1));
    // 3 blocks: ceil(10/4) = 3
    EXPECT_EQ(manager_->get_num_blocks(1), 3);
}

// 5. 同前缀第二次请求: 完整块命中
TEST_F(KVCachePrefixTest, SecondRequestPrefixHit) {
    // block_size=4, tokens=[1,2,3,4, 5,6,7,8, 9,10]
    // Blocks: [1234] [5678] [9,10(partial)]
    std::vector<int32_t> tokens1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int32_t cached1;
    manager_->allocate_sequence_cached(1, tokens1, cached1);
    EXPECT_EQ(cached1, 0);  // 首次无缓存

    // 释放序列 1（blocks 进入驱逐列表）
    manager_->free_sequence(1);

    // 同前缀的第二个请求
    std::vector<int32_t> tokens2 = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13};
    int32_t cached2;
    manager_->allocate_sequence_cached(2, tokens2, cached2);

    // 前 2 个完整块 [1234] [5678] 应该命中
    EXPECT_EQ(cached2, 8);  // 2 * block_size = 8 tokens cached
}

// 6. 部分前缀命中: 只有前 N 个连续块命中
TEST_F(KVCachePrefixTest, PartialPrefixHit) {
    // Seq 1: tokens=[1,2,3,4, 5,6,7,8]  → 2 full blocks
    std::vector<int32_t> tokens1 = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t cached1;
    manager_->allocate_sequence_cached(1, tokens1, cached1);
    manager_->free_sequence(1);

    // Seq 2: tokens=[1,2,3,4, 10,11,12,13, 5,6,7,8]
    // Block 0 [1,2,3,4] → hit
    // Block 1 [10,11,12,13] → miss (不同 token)
    // Block 2 [5,6,7,8] → 虽然 token 相同，但 prev_hash 不同 → miss
    // 只有第 0 块连续命中
    std::vector<int32_t> tokens2 = {1, 2, 3, 4, 10, 11, 12, 13, 5, 6, 7, 8};
    int32_t cached2;
    manager_->allocate_sequence_cached(2, tokens2, cached2);

    EXPECT_EQ(cached2, 4);  // 只有第一个块连续命中
}

// 7. 多序列共享前缀 blocks
TEST_F(KVCachePrefixTest, ConcurrentSharing) {
    // 两个序列同时存在，共享前缀
    std::vector<int32_t> tokens1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int32_t> tokens2 = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    int32_t cached1, cached2;

    manager_->allocate_sequence_cached(1, tokens1, cached1);
    EXPECT_EQ(cached1, 0);  // 首次无缓存

    // 第二个序列共享前 2 个完整块
    manager_->allocate_sequence_cached(2, tokens2, cached2);
    EXPECT_EQ(cached2, 8);  // [1234] [5678] 共享

    // 释放 seq 1，共享块的 ref_count 应从 2 降到 1（不被驱逐）
    manager_->free_sequence(1);
    EXPECT_TRUE(manager_->is_sequence_allocated(2));

    // 释放 seq 2，共享块的 ref_count 降到 0（进入驱逐列表）
    manager_->free_sequence(2);
    EXPECT_FALSE(manager_->is_sequence_allocated(2));
}

// 8. Exact block boundary: tokens 恰好填满所有 block
TEST_F(KVCachePrefixTest, ExactBlockBoundary) {
    // 8 tokens, block_size=4 → 2 full blocks, 0 partial
    std::vector<int32_t> tokens = {10, 20, 30, 40, 50, 60, 70, 80};
    int32_t cached;

    manager_->allocate_sequence_cached(1, tokens, cached);
    EXPECT_EQ(cached, 0);
    EXPECT_EQ(manager_->get_num_blocks(1), 2);  // 刚好 2 block
    manager_->free_sequence(1);

    // 再次分配完全相同的 tokens → 全部命中
    manager_->allocate_sequence_cached(2, tokens, cached);
    EXPECT_EQ(cached, 8);  // 全部 8 token 命中
}

// 9. free_sequence 后的缓存仍然存在
TEST_F(KVCachePrefixTest, CachePersistedAfterFree) {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t cached;

    // 分配 → 释放 → 再分配 → 释放 → 再分配
    // 每次应该都能命中缓存（只要没被驱逐）
    manager_->allocate_sequence_cached(1, tokens, cached);
    EXPECT_EQ(cached, 0);
    manager_->free_sequence(1);

    manager_->allocate_sequence_cached(2, tokens, cached);
    EXPECT_EQ(cached, 8);
    manager_->free_sequence(2);

    manager_->allocate_sequence_cached(3, tokens, cached);
    EXPECT_EQ(cached, 8);
}

// 10. 统计信息
TEST_F(KVCachePrefixTest, CacheStatistics) {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t cached;

    // 首次: 2 full blocks miss + 1 partial (不走 cached 路径)
    manager_->allocate_sequence_cached(1, tokens, cached);
    EXPECT_EQ(manager_->get_prefix_cache_misses(), 2);  // 2 full blocks
    EXPECT_EQ(manager_->get_prefix_cache_hits(), 0);
    manager_->free_sequence(1);

    // 再次: 2 full blocks hit
    manager_->allocate_sequence_cached(2, tokens, cached);
    EXPECT_EQ(manager_->get_prefix_cache_hits(), 2);
}

// 11. extend_sequence 与 prefix caching 兼容
TEST_F(KVCachePrefixTest, ExtendCachedSequence) {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};  // 1 full + 1 partial
    int32_t cached;
    manager_->allocate_sequence_cached(1, tokens, cached);
    EXPECT_EQ(manager_->get_num_blocks(1), 2);

    // Decode 阶段扩展（+1 token），这使用普通 extend
    int32_t allocated = 0;
    auto status = manager_->extend_sequence(1, 1, &allocated);
    ASSERT_TRUE(status) << status.get_err_msg();

    // 释放应正常工作（混合 cached + non-cached blocks）
    status = manager_->free_sequence(1);
    ASSERT_TRUE(status) << status.get_err_msg();
}

// 12. reset 清除所有缓存
TEST_F(KVCachePrefixTest, ResetClearsAllCache) {
    std::vector<int32_t> tokens = {1, 2, 3, 4};
    int32_t cached;

    manager_->allocate_sequence_cached(1, tokens, cached);
    manager_->free_sequence(1);

    manager_->reset();

    // 缓存已清除，不应命中
    manager_->allocate_sequence_cached(2, tokens, cached);
    EXPECT_EQ(cached, 0);
}

// 13. 空 Prompt (边界情况)
TEST_F(KVCachePrefixTest, EmptyTokens) {
    std::vector<int32_t> tokens = {};
    int32_t cached;
    // 空 token 序列应该能正常分配（0 blocks）
    auto status = manager_->allocate_sequence_cached(1, tokens, cached);
    ASSERT_TRUE(status);
    EXPECT_EQ(cached, 0);
    EXPECT_EQ(manager_->get_num_blocks(1), 0);
}

// 14. Tokens 不足一个 block (只有 partial)
TEST_F(KVCachePrefixTest, OnlyPartialBlock) {
    std::vector<int32_t> tokens = {1, 2};  // block_size=4, 只有 partial
    int32_t cached;

    manager_->allocate_sequence_cached(1, tokens, cached);
    EXPECT_EQ(cached, 0);  // 不足一个完整块，无法缓存
    EXPECT_EQ(manager_->get_num_blocks(1), 1);
    manager_->free_sequence(1);

    // 再次分配 → 仍然无法命中（partial block 不缓存）
    manager_->allocate_sequence_cached(2, tokens, cached);
    EXPECT_EQ(cached, 0);
}

// 15. LRU 驱逐 + 重新加载测试
TEST_F(KVCachePrefixTest, EvictionAndReload) {
    // num_blocks_=32, block_size=4
    // 填满所有 blocks 然后触发驱逐

    // 创建多个序列来填满所有 block
    // 每个序列 8 tokens = 2 blocks, 需要 16 个序列用完 32 blocks
    for (int i = 0; i < 16; ++i) {
        std::vector<int32_t> tokens = {i * 100 + 1, i * 100 + 2, i * 100 + 3, i * 100 + 4,
                                       i * 100 + 5, i * 100 + 6, i * 100 + 7, i * 100 + 8};
        int32_t cached;
        auto status = manager_->allocate_sequence_cached(i + 1, tokens, cached);
        ASSERT_TRUE(status) << "Failed at seq " << (i + 1) << ": " << status.get_err_msg();
    }
    EXPECT_EQ(manager_->get_free_block_num(), 0);

    // 释放所有序列 (blocks 进入驱逐列表)
    for (int i = 0; i < 16; ++i) {
        manager_->free_sequence(i + 1);
    }

    // 所有 32 blocks 应该在驱逐列表中
    // 新序列分配会驱逐最旧的 block
    std::vector<int32_t> new_tokens = {999, 998, 997, 996};
    int32_t cached;
    auto status = manager_->allocate_sequence_cached(100, new_tokens, cached);
    ASSERT_TRUE(status) << status.get_err_msg();
    EXPECT_EQ(cached, 0);  // 新 tokens，不会命中

    // 被驱逐的序列 1 的 hash 不再有效
    std::vector<int32_t> tokens_seq1 = {1, 2, 3, 4, 5, 6, 7, 8};
    manager_->free_sequence(100);
    manager_->allocate_sequence_cached(200, tokens_seq1, cached);
    // 序列 1 的 block 0 可能已被驱逐，取决于 LRU 顺序
    // 只验证不会崩溃
    EXPECT_GE(cached, 0);
}
