cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wnon-virtual-dtor")

add_executable(test_argparsecpp test_argparsecpp.cpp argparsecpp.cpp)
