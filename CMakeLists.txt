cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

project(qp_ipm VERSION 0.1.0)
add_executable(qp_ipm main.cpp src/ipm.cpp src/linesearch.cpp src/optimdata.cpp)

# add_compile_options(-Wall -Wextra -pedantic -Werror -m32)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(qp_ipm Eigen3::Eigen)

include_directories(src)
include_directories(lib)
include_directories(include)