/*
copied from https://github.com/hughperkins/coriander, April 5th 2022
*/

#include "mutations.h"

#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <iostream>

using namespace llvm;
using namespace std;


void mutateGlobalConstructorNumElements(GlobalVariable *var, int numElements) {
    Type *constructorElementType = cast<ArrayType>(var->getType()->getElementType())->getElementType();
    Type *newVartype = PointerType::get(ArrayType::get(constructorElementType, numElements), 0);
    var->mutateType(newVartype);
}

void appendGlobalConstructorCall(Module *M, std::string functionName) {
    GlobalVariable *ctors = cast<GlobalVariable>(M->getNamedValue("llvm.global_ctors"));
    int oldNumConstructors = cast<ArrayType>(cast<PointerType>(ctors->getType())->getElementType())->getNumElements();
    mutateGlobalConstructorNumElements(ctors, oldNumConstructors + 1);

    ConstantArray *initializer = cast<ConstantArray>(ctors->getInitializer());

    Constant **initializers = new Constant *[oldNumConstructors + 1];
    for(int i = 0; i < oldNumConstructors; i++) {
        initializers[i] = initializer->getAggregateElement((unsigned int)i);
    }
    Constant *structValues[] = {
        ConstantInt::get(IntegerType::get(M->getContext(), 32), 1000000),
        M->getOrInsertFunction(
            functionName,
            Type::getVoidTy(M->getContext()),
            static_cast<size_t>(NULL)),
        ConstantPointerNull::get(PointerType::get(IntegerType::get(M->getContext(), 8), 0))
    };
    initializers[oldNumConstructors] = ConstantStruct::getAnon(ArrayRef<Constant *>(&structValues[0], &structValues[3]));
    Constant *newinit = ConstantArray::get(initializer->getType(), ArrayRef<Constant *>(&initializers[0], &initializers[oldNumConstructors + 1]));
    ctors->setInitializer(newinit);
    delete[] initializers;
}

GlobalVariable *addGlobalVariable(Module *M, string name, string value) {
    int N = value.size() + 1;
    LLVMContext &context = M->getContext();
    ArrayType *strtype = ArrayType::get(IntegerType::get(context, 8), N);
    M->getOrInsertGlobal(StringRef(name), strtype);
    ConstantDataArray *constchararray = cast<ConstantDataArray>(ConstantDataArray::get(context, ArrayRef<uint8_t>((const uint8_t *)value.c_str(), N)));
    GlobalVariable *str = M->getNamedGlobal(StringRef(name));
    str->setInitializer(constchararray);
    return str;
}

Instruction *addStringInstr(Module *M, string name, string value) {
    // check if already exists first
    GlobalVariable *probe = M->getNamedGlobal(name);
    if(probe != 0) {
        return addStringInstrExistingGlobal(M, name);
    }

    GlobalVariable *var = addGlobalVariable(M, name, value);

    int N = value.size() + 1;
    LLVMContext &context = M->getContext();
    ArrayType *arrayType = ArrayType::get(IntegerType::get(context, 8), N);
    Value * indices[] = {
        ConstantInt::getSigned(IntegerType::get(context, 32), 0),
        ConstantInt::getSigned(IntegerType::get(context, 32), 0)
    };
    GetElementPtrInst *elem = GetElementPtrInst::CreateInBounds(arrayType, var, ArrayRef<Value *>(indices, 2));
    return elem;
}

Instruction *addStringInstrExistingGlobal(Module *M, string name) {
    GlobalVariable *var = M->getNamedGlobal(name);

    Type *varType = var->getType();
    if(ArrayType *arrayType1 = dyn_cast<ArrayType>(cast<PointerType>(varType)->getElementType())) {
        int N = arrayType1->getNumElements();

        LLVMContext &context = M->getContext();
        ArrayType *arrayType = ArrayType::get(IntegerType::get(context, 8), N);
        Value * indices[] = {
            ConstantInt::getSigned(IntegerType::get(context, 32), 0),
            ConstantInt::getSigned(IntegerType::get(context, 32), 0)
        };
        GetElementPtrInst *elem = GetElementPtrInst::CreateInBounds(arrayType, var, ArrayRef<Value *>(indices, 2));
        return elem;
    } else {
        throw runtime_error("unexpected type at addStringInstrExistingGlobal, not an arraytype");
    }
}

llvm::Constant *createInt32Constant(llvm::LLVMContext *context, int value) {
    return ConstantInt::getSigned(IntegerType::get(*context, 32), value);
}

void updateAddressSpace(Value *value, int newSpace) {
    if(PointerType *pointerType = dyn_cast<PointerType>(value->getType())) {
        Type *newType = PointerType::get(pointerType->getElementType(), newSpace);
        value->mutateType(newType);
    }
}

void copyAddressSpace(Value *src, Value *dest) {
    // copies address space from src value to dest value
    if(PointerType *srcType = dyn_cast<PointerType>(src->getType())) {
        if(isa<PointerType>(dest->getType())) {
            int addressspace = srcType->getAddressSpace();
            if(addressspace != 0) {
                updateAddressSpace(dest, addressspace);
            }
        }
    }
}
