#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("NanoInfer");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;

    testing::GTEST_FLAG(filter) = "SchedulerTest.*";

    FLAGS_v = 2;

    LOG(INFO) << "Start Test...\n";
    return RUN_ALL_TESTS();
}