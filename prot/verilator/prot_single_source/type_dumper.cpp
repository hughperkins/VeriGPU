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

#include "cocl/type_dumper.h"

#include "cocl/mutations.h"
#include "EasyCL/util/easycl_stringhelper.h"

#include "cocl/llvm_dump.h"

#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <set>

using namespace std;
using namespace llvm;
using namespace cocl;

namespace cocl {

// everything here assumes device-side
// (since hostside wouldnt be using opencl, just bytecode stuff)

// having said that, we have two types of struct declarations:
// - marshalling structs, which have pointers removed/transformed
// - what we will call 'deviceside' structs, which are the structs used in the kernel code, after
//.  any boilerplate etc
//
// This code handles only the 'deviceside' structs for now
// 'marshalling' structs are handled by struct_clone.cpp, on the whole

std::string TypeDumper::dumpAddressSpace(llvm::Type *type) {
    if(PointerType *ptr = dyn_cast<PointerType>(type)) {
        int addressspace = ptr->getAddressSpace();
        switch(addressspace) {
            case 0:
                return "";
            case 1:
                return "global";
            case 3:
                return "local";
            case 5:
                return "__vmem__";
            default:
                throw runtime_error("not implemented, addressspace " + easycl::toString(addressspace));
        }
    } else {
        return "";
    }
}

std::string TypeDumper::dumpPointerType(PointerType *ptr, bool decayArraysToPointer) {
    // std::cout << "dumpPointerType" << std::endl;
    string gencode = "";
    Type *elementType = ptr->getElementType();
    string elementTypeString = dumpType(elementType, decayArraysToPointer);
    int addressspace = ptr->getAddressSpace();
    if(addressspace == 5) {
        int depth = getPointerDepth(elementType);
        switch(depth) {
            case 0:
                return "__vmem__ unsigned long";
                break;
            case 1:
                return "__vmem__ unsigned long global *";
                break;
            default:
                std::cout << "not implemented depth=" << depth << std::endl;
                throw std::runtime_error("not implemenented depth");
        }
    }
    if(addressspace == 6) {
        int depth = getPointerDepth(elementType);
        std::cout << "dumpPointer type pointerdpeth=" << depth << std::endl;
        switch(depth) {
            case 1:
                return "__vmem2__ unsigned long";
                break;
            default:
                std::cout << "not implemented depth=" << depth << std::endl;
                throw std::runtime_error("not implemenented depth");
        }
    }
    string addressspacestr = "";
    switch(addressspace) {
        case 0:
            break;

        case 1:
            addressspacestr = "global";
            break;

        case 3:
            addressspacestr = "local";
            break;

        case 4:
            addressspacestr = "constant";
            break;

        case 5:
            addressspacestr = "__vmem__";
            break;

        case 6:
            addressspacestr = "__vmem2__";
            break;

        default:
            std::cout << "unimplemented addressspace " << addressspace << std::endl;
            throw std::runtime_error("Unimplemented addressspace");
    }

    // we're just going to hackily assume that anything that is `global **` should be `global * global *`
    // if(isa<PointerType>(ptr->getPointerElementType())) {
        // return "global " + elementTypeString + addressspacestr + " *";
        // return elementTypeString + addressspacestr + " *";
        // return "global " + elementTypeString + addressspacestr + " *";
    // }
    if(addressspacestr != "") {
        gencode += addressspacestr + " ";
    }
    gencode += elementTypeString + "*";
    return gencode;
}

std::string TypeDumper::dumpIntegerType(IntegerType *type) {
    switch(type->getPrimitiveSizeInBits()) {
        case 32:
            return "int";
        case 64:
            return "long";
        case 16:
            return "short";
        case 8:
            return "char";
        case 1:
            return "bool";
        default:
            outs() << "integer size " << type->getPrimitiveSizeInBits() << "\n";
            throw runtime_error("unrecognized size");
    }
}

std::string TypeDumper::addStructToGlobalNames(StructType *type) {
    if(globalNames->hasName(type)) {
        return globalNames->getName(type);
    }
    if(type->hasName()) {
        string name = type->getName();
        name = easycl::replaceGlobal(name, ".", "_");
        name = easycl::replaceGlobal(name, ":", "_");
        name = easycl::replaceGlobal(name, "<", "_");
        name = easycl::replaceGlobal(name, ">", "_");
        name = easycl::replaceGlobal(name, ", ", "_");
        if(name == "struct_float4") {
            name = "float4";
            name = globalNames->getOrCreateName(type, name);
            return "float4";
        } else {
            if(name.find("struct_") == 0 || name.find("struct ") == 0) {
                name[6] = ' ';
                name = globalNames->getOrCreateName(type, name);
                structsToDefine.insert(type);
                return name;
            } else if(name.find("class_") != string::npos) {
                name = "struct " + name;
                name = globalNames->getOrCreateName(type, name);
                structsToDefine.insert(type);
                return name;
            } else {
                name = "struct " + name;
                name = globalNames->getOrCreateName(type, name);
                structsToDefine.insert(type);
                return name;
            }
        }
    } else {
        COCL_LLVM_DUMP(type);
        throw runtime_error("not implemented: anonymous struct types");
    }
}

std::string TypeDumper::dumpStructType(StructType *type) {
    // std::cout << "TypeDumper::dumpStructType" << std::endl;
    return addStructToGlobalNames(type);
}

std::string TypeDumper::dumpArrayType(ArrayType *type, bool decayArraysToPointer) {
    // std::cout << "TypeDumper::dumpArrayType" << std::endl;
    ostringstream oss;
    int length = type->getNumElements();
    Type *elementType = type->getElementType();

    if(decayArraysToPointer) {
        oss << dumpType(elementType, decayArraysToPointer) + "*";
    } else {
        oss << dumpType(elementType, decayArraysToPointer) + "[" + easycl::toString(length) + "]";
    }
    return oss.str();
}

std::string TypeDumper::dumpVectorType(VectorType *vectorType, bool decayArraysToPointer) {
    // std::cout << "TypeDumper::dumpVectorType" << std::endl;
    int elementCount = vectorType->getNumElements();
    Type *elementType = vectorType->getElementType();
    if(elementType->getPrimitiveSizeInBits() == 0) {
        cout << endl;
        COCL_LLVM_DUMP(vectorType);
        cout << endl;
        throw runtime_error("TypeDumper::dumpVectorType: not implemented for non-primitive types");
    }
    ostringstream oss;
    oss << dumpType(elementType);
    if(decayArraysToPointer) {
        oss << "*";
    } else {
        oss << "[" << elementCount << "]";
    }
    cout << oss.str();
    return oss.str();
}

std::string TypeDumper::dumpFunctionType(FunctionType *fn) {
    std::string params_str = "";
    int i = 0;
    for(auto it=fn->param_begin(); it != fn->param_end(); it++) {
        Type * paramType = *it;
        if(i > 0) {
            params_str += ", ";
        }
        params_str += dumpType(paramType);
        i++;
    }
    return params_str;
}

std::string TypeDumper::dumpType(Type *type, bool decayArraysToPointer) {
    // std::cout << "  dumpType()" << std::endl;
    Type::TypeID typeID = type->getTypeID();
    switch(typeID) {
        case Type::VoidTyID:
            return "void";

        case Type::FloatTyID:
            return "float";

        // case Type::UnionTyID:
        //     throw runtime_error("not implemented: union type");

        case Type::StructTyID:
            return dumpStructType(cast<StructType>(type));

        case Type::VectorTyID:
            return dumpVectorType(cast<VectorType>(type));

        case Type::ArrayTyID:
            return dumpArrayType(cast<ArrayType>(type), decayArraysToPointer);

        case Type::DoubleTyID:
            if(forceSingle) {
                return "float";
            }
            return "double";

        case Type::FunctionTyID:
            return dumpFunctionType(cast<FunctionType>(type));

        case Type::PointerTyID:
            return dumpPointerType(cast<PointerType>(type), decayArraysToPointer);

        case Type::IntegerTyID:
            return dumpIntegerType(cast<IntegerType>(type));

        default:
            outs() << "type id " << typeID << "\n";
            cout << endl;
            COCL_LLVM_DUMP(type);
            cout << endl;
            throw runtime_error("TypeDumper::dumpType(...): unrecognized type");
    }
}

int TypeDumper::getPointerDepth(Type *type) {
    // std::cout << " getPointerDepth()" << std::endl;
    if(PointerType *nextLevel = dyn_cast<PointerType>(type)) {
        return getPointerDepth(nextLevel->getElementType()) + 1;
    }
    return 0;
}

std::string TypeDumper::dumpStructDefinition(StructType *type, string name) {
    // dump deviceside struct opencl
    // this will write pointers as global pointers, probably

    // std::cout << "TypeDumper::dumpStructDefinition" << std::endl;
    ostringstream declaration;
    declaration << name << " {\n";
    int i = 0;
    for(auto it=type->element_begin(); it != type->element_end(); it++) {
        Type *elementType = *it;
        bool isPtrPtrFunction = false;
        if(PointerType *ptr = dyn_cast<PointerType>(elementType)) {
            Type *ptrElementType = ptr->getElementType();
            if(PointerType *ptrptr = dyn_cast<PointerType>(ptrElementType)) {
                Type *ptrptrElementType = ptrptr->getElementType();
                isPtrPtrFunction = isa<FunctionType>(ptrptrElementType);
            }
        }
        if(isPtrPtrFunction) {
            // we can ignore functions, whilst dumping structs
            continue;
        }
        std::string memberName = "f" + easycl::toString(i);
        if(ArrayType *arraytype = dyn_cast<ArrayType>(elementType)) {
            Type *arrayelementtype = arraytype->getElementType();
            int numElements = arraytype->getNumElements();
            declaration << "    ";
            // assume any pointers are global
            // also, we'll change them into unsigned longs ... :-P. (ie virtual mem locations)
            if(isa<PointerType>(arrayelementtype)) {
                declaration << "__vmem__ unsigned long ";
            } else {
                declaration << dumpType(arrayelementtype) << " ";
            }
            declaration << memberName << "[" << numElements << "];\n";
        } else {
            declaration << "    ";
            // if its a pointer:
            // - level 1 => global (we'll splice it in)
            // - level 2 => vmem array
            if(isa<PointerType>(elementType)) {
                int pointerDepth = getPointerDepth(elementType);
                if(pointerDepth > 1) {
                    declaration << "__vmem2__ unsigned long " << memberName << ";\n";
                } else {
                    declaration << "global " << dumpType(elementType) << " " << memberName << ";\n";
                }
            } else {
                declaration << dumpType(elementType) << " " << memberName << ";\n";
            }
        }
        i++;
    }
    declaration << "};\n";

    return declaration.str();
}

std::string TypeDumper::dumpStructDefinitions() {
    string gencode = "";
    set<StructType *>dumped;
    while(dumped.size() < structsToDefine.size()) {
        for(auto it=structsToDefine.begin(); it != structsToDefine.end(); it++) {
            StructType *structType = *it;
            if(dumped.find(structType) != dumped.end()) {
                continue;
            }
            // check if we already defined its members
            bool dumpable = true;
            for(auto it2=structType->element_begin(); it2 != structType->element_end(); it2++) {
                Type *subType = (*it2);
                if(ArrayType *elementArrayType = dyn_cast<ArrayType>(subType)) {
                    subType = elementArrayType->getElementType();
                }
                if(StructType *elementStructType = dyn_cast<StructType>(subType)) {
                    if(dumped.find(elementStructType) == dumped.end()) {
                        structsToDefine.insert(elementStructType);
                        dumpable = false;
                        break;
                    } else {
                    }
                }
            }
            if(!dumpable) {
                continue;
            }
            dumped.insert(structType);
            addStructToGlobalNames(structType);
            gencode += dumpStructDefinition(structType, globalNames->getName(structType));
        }
    }
    return gencode;
}

} // namespace cocl;
