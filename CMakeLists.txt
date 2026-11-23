cmake_minimum_required(VERSION 3.18)
project(cuda_llm_kernel_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(llm_kernel_bench
    src/main.cu
    src/kernels/rmsnorm.cu
    src/kernels/softmax.cu
    src/kernels/sgemm.cu
)

target_include_directories(llm_kernel_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(llm_kernel_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)
target_link_libraries(llm_kernel_bench PRIVATE CUDA::cudart CUDA::cublas)
