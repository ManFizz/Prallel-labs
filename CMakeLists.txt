cmake_minimum_required(VERSION 3.25)
project(PP_Labs)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(PP_Lab_1 lib.cpp 1.cpp barrier.cpp)
add_executable(PP_Lab_2 lib.cpp 2.cpp barrier.cpp)

target_link_libraries(PP_Lab_1 PRIVATE pthread Threads::Threads OpenMP::OpenMP_CXX)
target_link_libraries(PP_Lab_2 PRIVATE pthread Threads::Threads OpenMP::OpenMP_CXX)
