cmake_minimum_required(VERSION 3.16.3)

if(APPLE)
set(CLANG_HOME "/usr/local/opt/llvm-14.0.0" CACHE PATH "the downloaded clang+llvm-14.0.0-x86_64-apple-darwin folder, containing lib, bin etc")
# set(PYTHON27_PATH "/usr/bin/python2" CACHE FILEPATH "full path of python27")
elseif(WIN32)
# set(CLANG_HOME "C:\\Program Files\\LLVM" CACHE PATH "the downloaded clang-4.0.0 folder, containing lib, bin etc")
# set(PYTHON27_PATH "C:\\python27\\python" CACHE FILEPATH "full path of python27")
else()
set(CLANG_HOME "/usr/local/opt/llvm-14.0.0" CACHE PATH "the downloaded clang-144.0.0 folder, containing lib, bin etc")
# set(PYTHON27_PATH "/usr/bin/python2" CACHE FILEPATH "full path of python27")
endif()

set(CMAKE_CXX_COMPILER ${CLANG_HOME}/bin/clang++)
set(CMAKE_C_COMPILER ${CLANG_HOME}/bin/clang)

project(verigpu_runtime)

find_package(LLVM REQUIRED CONFIG PATHS "${CLANG_HOME}/lib/cmake/llvm")

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

llvm_map_components_to_libnames(llvm_libs support core irreader transformutils)

message(STATUS "llvm_libs ${llvm_libs}")

find_package(verilator HINTS $ENV{VERILATOR_ROOT})
# set(BASE CACHE STRING "root of verigpu repo" "${CMAKE_SOURCE_DIR}/../../..")
set(BASE "${CMAKE_SOURCE_DIR}/../../..")
set(SRC "${BASE}/src")
message(STATUS "BASE ${BASE}")

add_library(verigpu_runtime SHARED
    ${VERILATOR_ROOT}/include/verilated.cpp
    ${CMAKE_SOURCE_DIR}/gpu_runtime.cpp
    # ${CMAKE_SOURCE_DIR}/my_gpu_test_client.cpp
)
verilate(verigpu_runtime
    SOURCES
        ${CMAKE_SOURCE_DIR}/controller.sv
    TOP_MODULE controller
    PREFIX controller
)
target_compile_options(verigpu_runtime PRIVATE "-std=c++14")

# some of this stuff is copied from Coriander, apr 6 2022

execute_process(
    COMMAND ${BASE}/cmake/get-llvm-cxxflags.sh ${CLANG_HOME}
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "LLVM_CXXFLAGS ${LLVM_CXXFLAGS}")


# STRING(REGEX REPLACE " " ";" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
# string(REGEX REPLACE ";-std=c\\+\\+0x" ";-std=c++14" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
# string(REGEX REPLACE ";-fno-exceptions" ";-fexceptions" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
# string(REGEX REPLACE ";-DNDEBUG" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
# string(REGEX REPLACE ";-isysroot;[^;]" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

# # split LLVM_CXXFLAGS into defines vs other options
# STRING(REGEX MATCHALL ";-D[^;]+" LLVM_DEFINES "${LLVM_CXXFLAGS}")
# STRING(REGEX REPLACE ";+" ";" LLVM_DEFINES "${LLVM_DEFINES}")
# STRING(REGEX REPLACE ";-D[^;]+" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

# syslibs is non-llvm libs, used by llvm, eg -lz -lm -curses
# execute_process(
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm-syslibs.sh ${CLANG_HOME}
#     OUTPUT_VARIABLE LLVM_SYSLIBS
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# # llvm-libs is the actual llvm libs, like LLVMCore, etc
# execute_process(
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-llvm-libs.sh ${CLANG_HOME}
#     OUTPUT_VARIABLE LLVM_LIBNAMES
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# separate_arguments(LLVM_LIBNAMES)
# separate_arguments(LLVM_SYSLIBS)

# at this point, LLVM_LIBNAMES should be semi-colon separated, comprising llvm libfile names, without
# paths. We will add the path now:
# output of this bit is that LLVM_LIBPATHS should be semi-colon separated, and each item
# should be a full filepath of one llvm link library
# foreach(LIBNAME ${LLVM_LIBNAMES})
#     set(LLVM_LIBPATHS ${LLVM_LIBPATHS} ${CLANG_HOME}/lib/${LIBNAME})
# endforeach()

add_executable(patch_hostside
    ${CMAKE_SOURCE_DIR}/patch_hostside.cpp
    ${CMAKE_SOURCE_DIR}/mutations.cpp
    ${CMAKE_SOURCE_DIR}/readIR.cpp
    ${BASE}/third_party/argparsecpp/argparsecpp.cpp
    # src/type_dumper.cpp
    # src/GlobalNames.cpp
    # src/cocl_logging.cpp
)
target_include_directories(patch_hostside PRIVATE ${CLANG_HOME}/include)
# target_include_directories(patch_hostside PRIVATE include)
target_include_directories(patch_hostside PRIVATE ${BASE}/third_party/argparsecpp)
# target_include_directories(patch_hostside PRIVATE src/EasyCL)
target_compile_options(patch_hostside PRIVATE ${LLVM_CXXFLAGS})
target_link_libraries(patch_hostside "${LLVM_LIBPATHS}" ${llvm_libs})

# should try running the following now:
# patch_hostside --devicellfile ~/git/toy_proc/examples/cpp_single_source/sum_ints/build_bash/sum_ints.ll \
#     --hostrawfile ~/git/toy_proc/examples/cpp_single_source/sum_ints/build_bash/sum_ints-hostraw.ll \
#     --hostpatchedfile foo.ll


# --devicellfile ~/git/toy_proc/examples/cpp_single_source/sum_ints/build_bash/sum_ints.ll --hostrawfile ~/git/toy_proc/examples/cpp_single_source/sum_ints/build_bash/sum_ints-hostraw.ll --hostpatchedfile foo.ll
