# Builds the optional replay kernel. The kernel is a development and CI
# artifact: the Python package never requires it, and `train-guard sweep`
# falls back to the pure-Python engine when the binary is absent.
#
# The floating-point flags are part of the correctness contract: the
# kernel must reproduce CPython double arithmetic bit for bit, so FMA
# contraction stays off and no fast-math variant is ever acceptable.

cmake_minimum_required(VERSION 3.16)
project(train_guard_kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_executable(train-guard-kernel replay_kernel.cpp)
target_link_libraries(train-guard-kernel PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(train-guard-kernel PRIVATE /O2 /fp:precise /W4)
else()
  target_compile_options(train-guard-kernel PRIVATE -O2 -ffp-contract=off -Wall -Wextra)
endif()
