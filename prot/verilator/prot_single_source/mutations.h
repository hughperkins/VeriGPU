/*
Copied from https://github.com/hughperkins/coriander, april 5 2022
*/

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"

#include <string>

void mutateGlobalConstructorNumElements(llvm::GlobalVariable *var, int numElements);
void appendGlobalConstructorCall(llvm::Module *M, std::string functionName);
llvm::GlobalVariable *addGlobalVariable(llvm::Module *M, std::string name, std::string value);
llvm::Instruction *addStringInstr(llvm::Module *M, std::string name, std::string value);
llvm::Instruction *addStringInstrExistingGlobal(llvm::Module *M, std::string name);
llvm::Constant *createInt32Constant(llvm::LLVMContext *context, int value);
void updateAddressSpace(llvm::Value *value, int newSpace);
void copyAddressSpace(llvm::Value *src, llvm::Value *dest);
