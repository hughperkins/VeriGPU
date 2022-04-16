#pragma once

#include <string>
#include <cstdint>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

class ReadIR {
public:
    static uint64_t readIntConstant_uint64(llvm::ConstantInt *constant);
    static uint32_t readIntConstant_uint32(llvm::ConstantInt *constant);
    static std::string getName(llvm::StructType *type);
    static std::string getName(llvm::Function *type);
    static std::string getName(llvm::Value *value);
    static int readInt32Constant(llvm::Value *value);
    static float readFloatConstant(llvm::Value *value);
    static std::string dumpFloatConstant(bool forceSingle, llvm::ConstantFP *constantFP);
};
