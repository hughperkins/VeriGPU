// Copyright Hugh Perkins 2016

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "readIR.h"
#include "llvm_dump.h"

#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <iostream>
#include <sstream>

using namespace std;
using namespace llvm;

uint64_t ReadIR::readIntConstant_uint64(ConstantInt *constant) {
    return constant->getZExtValue();
}

uint32_t ReadIR::readIntConstant_uint32(ConstantInt *constant) {
    assert(constant->getBitWidth() <= 32);
    return (uint32_t)constant->getZExtValue();
}

std::string ReadIR::getName(StructType *type) {
    if(!type->hasName()) {
        COCL_LLVM_DUMP(type);
        throw runtime_error("type doesnt have name");
    }
    return type->getName().str();
}

std::string ReadIR::getName(Function *type) {
    if(!type->hasName()) {
        type->print(llvm::dbgs());
        //COCL_LLVM_DUMP(type);
        throw runtime_error("function doesnt have name");
    }
    return type->getName().str();
}

std::string ReadIR::getName(Value *value) {
    if(!value->hasName()) {
        COCL_LLVM_DUMP(value);
        throw runtime_error("value doesnt have name");
    }
    return value->getName().str();
}

int ReadIR::readInt32Constant(Value *value) {
    return cast<ConstantInt>(value)->getSExtValue();
}

string ReadIR::dumpFloatConstant(bool forceSingle, ConstantFP *constantFP) {
    double doubleValue;
    float floatValue;
    bool isDouble = false;
    ostringstream oss;
    const APFloat *apf = &constantFP->getValueAPF();
    switch(constantFP->getType()->getTypeID()) {
        case Type::FloatTyID:
            floatValue = apf->convertToFloat();
            oss << floatValue;
            break;
        case Type::DoubleTyID:
            isDouble = true;
            doubleValue = apf->convertToDouble();
            oss << doubleValue;
            break;
        default:
            throw runtime_error("unrecognized type");
    }
//         oss << floatvalue;
    string valuestr = oss.str();
    if(valuestr == "inf") {
        return "INFINITY";
    } else if(valuestr == "-inf") {
        return "-INFINITY";
    }
    float asFloat = readFloatConstant(constantFP);
    oss.str("");
    oss << asFloat;
    valuestr = oss.str();
    if(valuestr == "inf") {
        return "INFINITY";
    } else if(valuestr == "-inf") {
        return "-INFINITY";
    }

    // check for scientific notation, and lack of decimal point. if neither, add ".0"
    if(valuestr.find('.') == string::npos && valuestr.find("e") == string::npos) {
        valuestr += ".0";
    }
    if(!isDouble || forceSingle) {
        valuestr += "f";
    }
    return valuestr;
}

float ReadIR::readFloatConstant(Value *value) {
    // cout << endl;
    // cout << "isa double " << (value->getType()->getTypeID() == Type::DoubleTyID) << endl;
    // cout << "isa float " << (value->getType()->getTypeID() == Type::FloatTyID) << endl;
    const APFloat *apf = &cast<ConstantFP>(value)->getValueAPF();
    // cout << "is negative " << apf->isNegative() << " isinfinity " << apf->isInfinity() << " isLargest " << apf->isLargest() << endl;
    // double valueasdouble = apf->convertToDouble();
    // float valueasfloat = apf->convertToFloat();
    // float valueasdoubletofloat = (float)valueasdouble;
    // if(apf->isInfinity()) {
    //     if(apf->isNegative()) {

    //     }
    // }
    float res = 0;
    switch(value->getType()->getTypeID()) {
        case Type::FloatTyID:
            res = apf->convertToFloat();
            break;
        case Type::DoubleTyID:
            res = (float)apf->convertToDouble();
            break;
        default:
            throw runtime_error("unrecognized type");
    }
    // cout << "readfloatConstant returning " << res << endl;
    return res;
    // cout << "float constant as double " << valueasdouble << endl;
    // cout << "float constant as float " << valueasfloat << endl;
    // cout << "float constant as float, via double " << valueasdoubletofloat << endl;
    // cout << "res " << res << endl;
    // return valueasdoubletofloat;
}
