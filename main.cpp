#include <glog/logging.h>
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path ";
        return -1;
    }

    return 0;
}