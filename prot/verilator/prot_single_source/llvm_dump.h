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

/// \author Aksel Alpay

#ifndef COCL_LLVM_DUMP_H
#define COCL_LLVM_DUMP_H

#include <llvm/Support/Debug.h>

#if LLVM_VERSION_MAJOR > 4
#define COCL_LLVM_DUMP(entity) (entity)->print(llvm::dbgs(), true)
#else
#define COCL_LLVM_DUMP(entity) ((entity)->dump())
#endif

#endif

