add_executable(llama2_infer llama2.cpp)
target_link_directories(llama2_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama2_infer llama)

set_target_properties(llama2_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
