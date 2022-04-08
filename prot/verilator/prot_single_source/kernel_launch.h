// Copyright Hugh Perkins 2016, 2017

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// things we dont want other projects to need to include, eg tensorflow
// they'll just need to include coriander includes, not have eg llvm includes

#include <vector>

// #include "cocl/cocl_launch_args.h"
#include "kernel_launch_ext.h"
#include "launch_args.h"

// namespace VeriGPU
// {
//     class CLKernel;
//     class EasyCL;
// }

namespace VeriGPU
{
    // class CoclStream;

    // struct GenerateOpenCLResult
    // {
    //     std::string clSourcecode;
    //     std::string originalKernelName;
    //     std::string shortKernelName;
    //     std::string uniqueKernelName;
    // };
    // GenerateOpenCLResult generateOpenCL(int uniqueClmemCount, std::vector<int> &clmemIndexByClmemArgIndex, std::string origKernelName, std::string devicellsourcecode);
    // easycl::CLKernel *compileOpenCLKernel(std::string originalKernelName, std::string uniqueKernelName, std::string shortKernelName, std::string clSourcecode);
    // easycl::CLKernel *compileOpenCLKernel(std::string shortKernelName, std::string clSourcecode);

    class LaunchConfiguration
    {
    public:
        size_t grid[3];
        size_t block[3];
        // easycl::CLQueue *queue = 0;       // NOT owned by us
        // cocl::CoclStream *coclStream = 0; // NOT owned

        std::vector<std::unique_ptr<Arg>> args;

        // std::map<cl_mem, int> clmemIndexByClmem;
        // std::vector<cl_mem> clmems;
        // std::vector<int> clmemIndexByClmemArgIndex;

        // std::vector<cl_mem> kernelArgsToBeReleased;
        std::string kernelName = "";
        std::string uniqueKernelName = "";
        std::string shortKernelName = "";
        std::string devicellsourcecode = "";
    };
}
