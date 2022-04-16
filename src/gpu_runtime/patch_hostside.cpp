/*
This needs to take the -hostraw.ll file, and patch the calls to the kernel to instead
make calls to our gpu api

This is based on patch_hostside.cpp from https://github.com/hughperkins/coriander

We're going to comment out anything to do with structs for now, keep things simple
*/
// For doc, please see the corresponding include file, patch_hostside.h

#include "patch_hostside.h"

// #include "cocl/cocl_logging.h"
#include "llvm_dump.h"
#include "type_dumper.h"

#include "mutations.h"
#include "argparsecpp.h"
#include "stringhelper.h"

#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm/Analysis/LoopAccessAnalysis.h"

#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <regex>

using namespace llvm;
using namespace std;
using namespace veriGPU;

namespace veriGPU
{

    static llvm::LLVMContext context;
    static std::string deviceriscvcode_stringname;
    static string devicellfilename;
    static string deviceriscvfilename;

        std::string nameRemoveDeviceStub(std::string deviceStubName)
        {
            // e.g deviceStubName _Z23__device_stub__sum_intsPjjS_
            // returned name _Z8sum_intsPjjS_
            // we remove __device_stub__, and change the
            // count at the start of the name
            static const std::regex re("(_[^0-9]+)([0-9]+)__device_stub__(.*)");
            std::smatch matches;
            int ret = std::regex_search(deviceStubName, matches, re);
            int nameLen = std::stoi(matches[2]);
            int newLen = nameLen - strlen("__device_stub__");
            std::string newName = matches.str(1) + std::to_string(newLen) + matches.str(3);
            return newName;
        }

    // static GlobalNames globalNames;
    // static TypeDumper typeDumper(&globalNames);
    // static StructCloner structCloner(&typeDumper, &globalNames);

    // this should probably be more of an instance variable probably?
    static std::unique_ptr<LaunchCallInfo> launchCallInfo(new LaunchCallInfo);

    std::unique_ptr<GenericCallInst> GenericCallInst::create(llvm::InvokeInst *inst)
    {
        return unique_ptr<GenericCallInst>(new GenericCallInst_Invoke(inst));
    }
    std::unique_ptr<GenericCallInst> GenericCallInst::create(llvm::CallInst *inst)
    {
        return unique_ptr<GenericCallInst>(new GenericCallInst_Call(inst));
    }

    template <typename... ArgsTy>
    Function *getOrInsertFunction(Module *m, StringRef Name, Type *RetTy, ArgsTy... Args)
    {
        return cast<Function>(m->getOrInsertFunction(Name, RetTy, Args...).getCallee());
    }

    std::ostream &operator<<(std::ostream &os, const LaunchCallInfo &info)
    {
        raw_os_ostream my_raw_os_ostream(os);
        my_raw_os_ostream << "LaunchCallInfo " << info.kernelName;
        my_raw_os_ostream << "<<<";

        my_raw_os_ostream << ">>>";
        my_raw_os_ostream << "(";
        int i = 0;
        for (auto it = info.params.begin(); it != info.params.end(); it++)
        {
            if (i > 0)
            {
                my_raw_os_ostream << ", ";
            }
            const ParamInfo *paramInfo = it->get();
            paramInfo->typeDevicesideFn->print(my_raw_os_ostream);
            i++;
        }
        my_raw_os_ostream << ");\n";
        my_raw_os_ostream << "value types: ";
        i = 0;
        for (auto it = info.params.begin(); it != info.params.end(); it++)
        {
            const ParamInfo *paramInfo = it->get();
            Value *value = paramInfo->value;
            if (i > 0)
            {
                my_raw_os_ostream << ", ";
            }
            // my_raw_os_ostream << typeDumper.dumpType(value->getType());
            i++;
        }
        return os;
    }

    // std::ostream &operator<<(std::ostream &os, const RuntimePointerChecking::PointerInfo &pointerInfo)
    // {
    //     os << "PointerInfo(offset=" << pointerInfo.offset << ", type=" << typeDumper.dumpType(pointerInfo.type);
    //     os << " indices=";
    //     int i = 0;
    //     for (auto it = pointerInfo.indices.begin(); it != pointerInfo.indices.end(); it++)
    //     {
    //         if (i > 0)
    //         {
    //             os << ",";
    //         }
    //         os << *it;
    //         i++;
    //     }
    //     os << ")";
    //     return os;
    // }

    llvm::Instruction *PatchHostside::addSetKernelArgInst_int(llvm::Instruction *lastInst, llvm::Value *value, llvm::IntegerType *intType)
    {
        Module *M = lastInst->getModule();

        int bitLength = intType->getBitWidth();
        string mangledName = "";
        if (bitLength == 32)
        {
            mangledName = "setKernelArgInt32";
        }
        else if (bitLength == 64)
        {
            mangledName = "setKernelArgInt64";
        }
        else if (bitLength == 8)
        {
            mangledName = "setKernelArgInt8";
        }
        else
        {
            throw std::runtime_error("bitlength " + VeriGPU::toString(bitLength) + " not implemented");
        }

        Function *setKernelArgInt = cast<Function>(getOrInsertFunction(
            M,
            mangledName,
            Type::getVoidTy(context),
            IntegerType::get(context, bitLength)));

        CallInst *call = CallInst::Create(setKernelArgInt, value);
        call->insertAfter(lastInst);
        lastInst = call;
        return lastInst;
    }

    llvm::Instruction *PatchHostside::addSetKernelArgInst_float(llvm::Instruction *lastInst, llvm::Value *value)
    {
        // handle primitive floats, which we pass to `setKernelArgFloat`, by value

        Module *M = lastInst->getModule();

        Type *valueType = value->getType();
        if (valueType->isDoubleTy())
        {
            cout << "Executing functions with doubles as kernel parameters is not supported" << endl;
            cout << "Note that this is not set-in-stone, but is really hard to do." << endl;
            cout << "If you really need this functionality, please post your use-case into https://github.com/hughperkins/coriander/issues/22" << endl;
            throw runtime_error("Executing functions with doubles as kernel parameters is not supported");
        }

        Function *setKernelArgFloat = cast<Function>(getOrInsertFunction(
            M,
            "setKernelArgFloat",
            Type::getVoidTy(context),
            Type::getFloatTy(context)));

        CallInst *call = CallInst::Create(setKernelArgFloat, value);
        call->insertAfter(lastInst);
        return call;
    }

    llvm::Instruction *PatchHostside::addSetKernelArgInst_pointer(llvm::Instruction *lastInst, llvm::Value *value)
    {
        // Indentor indentor;
        Type *valueType = value->getType();
        Module *M = lastInst->getModule();

        Type *elementType = cast<PointerType>(valueType)->getElementType();
        if (elementType->isDoubleTy())
        {
            cout << "Executing functions with double arrays as kernel parameters is not supported" << endl;
            cout << "Note that this is not set-in-stone, but is really hard to do." << endl;
            cout << "If you really need this functionality, please post your use-case into https://github.com/hughperkins/coriander/issues/22" << endl;
            throw runtime_error("Executing functions with double arrays as kernel parameters is not supported");
        }

        BitCastInst *bitcast = new BitCastInst(value, PointerType::get(IntegerType::get(context, 8), 0));
        bitcast->insertAfter(lastInst);
        lastInst = bitcast;

        const DataLayout *dataLayout = &M->getDataLayout();
        int allocSize = dataLayout->getTypeAllocSize(elementType);
        int32_t elementSize = allocSize;

        Function *setKernelArgPointerVoid = cast<Function>(getOrInsertFunction(
            M,
            "setKernelArgPointerVoid",
            Type::getVoidTy(context),
            PointerType::get(IntegerType::get(context, 8), 0)));

        Value *args[] = {bitcast};
        CallInst *call = CallInst::Create(setKernelArgPointerVoid, ArrayRef<Value *>(args));
        call->insertAfter(lastInst);
        lastInst = call;
        return lastInst;
    }

    // llvm::Instruction *PatchHostside::addSetKernelArgInst_pointerstruct(llvm::Instruction *lastInst, llvm::Value *structPointer)
    // {
    //     // what this will need to do is:
    //     // - create a call to pass the gpu buffer, that contains the struct, to hostside_opencl_funcs, at runtime
    //     // we're going to forbid gpuside buffers containing pointers, for now

    //     // lets deal with the gpuside buffer first, that should be fairly easy-ish:
    //     lastInst = addSetKernelArgInst_pointer(lastInst, structPointer);

    //     // we'd better at least assert or something, if there are pointers in the struct
    //     Module *M = lastInst->getModule();
    //     Type *valueType = structPointer->getType();
    //     StructType *structType = cast<StructType>(cast<PointerType>(valueType)->getElementType());
    //     unique_ptr<StructInfo> structInfo(new StructInfo());
    //     StructCloner::walkStructType(M, structInfo.get(), 0, 0, vector<int>(), "", structType);
    //     bool structHasPointers = structInfo->pointerInfos.size() > 0;
    //     if (structHasPointers)
    //     {
    //         cout << "ERROR: Coriander currently forbids pointers inside gpu-side structs, passed into kernels" << endl;
    //         cout << "If you need this, please create an issue at https://github.com/hughperkins/Coriander/issues" << endl;
    //         cout << "Note that it's pretty hard to do though" << endl;
    //         // COCL_LLVM_DUMP(structPointer);
    //         cout << endl;
    //         throw std::runtime_error("ERROR: Coriander currently forbids pointers inside gpu-side structs, passed into kernels");
    //     }
    //     return lastInst;
    // }

    llvm::Instruction *PatchHostside::addSetKernelArgInst_byvaluevector(llvm::Instruction *lastInst, llvm::Value *vectorPointer)
    {
        // so, lets assume its just a hostside blob, that we need to send deviceside, as a blob
        // we should thus find out the blob size
        // we'll need:
        // - element count
        // - element size

        Type *valueType = vectorPointer->getType();
        VectorType *vectorType = cast<VectorType>(cast<PointerType>(valueType)->getElementType());
        int elementCount = vectorType->getElementCount().getFixedValue();
        Type *elementType = vectorType->getElementType();
        int elementSizeBytes = elementType->getPrimitiveSizeInBits() / 8;
        if (elementType->getPrimitiveSizeInBits() == 0)
        {
            cout << endl;
            // COCL_LLVM_DUMP(vectorType);
            cout << endl;
            throw runtime_error("PatchHostside::addSetKernelArgInst_byvaluevector: not implemented for non-primitive types");
        }

        int allocSize = elementSizeBytes * elementCount;

        Module *M = lastInst->getModule();

        BitCastInst *bitcast = new BitCastInst(vectorPointer, PointerType::get(IntegerType::get(context, 8), 0));
        bitcast->insertAfter(lastInst);
        lastInst = bitcast;

        Value *args[2];
        args[0] = bitcast;
        args[1] = createInt32Constant(&context, allocSize);

        Function *setKernelArgHostsideBuffer = cast<Function>(getOrInsertFunction(
            M,
            "setKernelArgHostsideBuffer",
            Type::getVoidTy(context),
            PointerType::get(IntegerType::get(context, 8), 0),
            IntegerType::get(context, 32)));

        CallInst *call = CallInst::Create(setKernelArgHostsideBuffer, ArrayRef<Value *>(args));
        call->insertAfter(lastInst);
        lastInst = call;

        return lastInst;
    }

    void addMetadata(Instruction *value, string tag)
    {
        MDNode *mdnode = MDNode::get(context, MDString::get(context, tag));
        value->setMetadata(tag, mdnode);
    }

//     llvm::Instruction *PatchHostside::addSetKernelArgInst_byvaluestruct(llvm::Instruction *lastInst, ParamInfo *paramInfo, llvm::Value *structPointer)
//     {
//         //
//         // what this is going to do is:
//         // - create a call to pass the hostside buffer to hostside_opencl_funcs, at runtime
//         // - if the struct contains pointers, then add appropriate calls to pass those at runtime too
//         //

//         Module *M = lastInst->getModule();

//         Indentor indentor;
//         PointerType *structPointerType = cast<PointerType>(structPointer->getType());
//         if (!isa<StructType>(structPointerType->getElementType()))
//         {
//             BitCastInst *bitcast = new BitCastInst(structPointer, paramInfo->typeDevicesideFn);
//             bitcast->insertAfter(lastInst);
//             structPointer = bitcast;
//             lastInst = bitcast;

//             structPointerType = cast<PointerType>(structPointer->getType());
//         }

//         StructType *structType = cast<StructType>(structPointerType->getElementType());
//         string typeName = structType->getName().str();
//         if (typeName == "struct.float4")
//         {
//             return PatchHostside::addSetKernelArgInst_pointer(lastInst, structPointer);
//         }

//         unique_ptr<StructInfo> structInfo(new StructInfo());
//         StructCloner::walkStructType(M, structInfo.get(), 0, 0, vector<int>(), "", structType);
//         bool structHasPointers = structInfo->pointerInfos.size() > 0;

//         string name = globalNames.getOrCreateName(structType);
//         Type *newType = structCloner.cloneNoPointers(structType);

//         const DataLayout *dataLayout = &M->getDataLayout();
//         int allocSize = dataLayout->getTypeAllocSize(newType);

//         Value *sourceStruct = 0;
//         if (structHasPointers)
//         {
// #if LLVM_VERSION_MAJOR > 4
//             // Assume address space 0
//             Instruction *alloca = new AllocaInst(newType, 0, "newalloca");
// #else
//             Instruction *alloca = new AllocaInst(newType, "newalloca");
// #endif
//             alloca->insertAfter(lastInst);
//             lastInst = alloca;

//             lastInst = structCloner.writeHostsideIrCopyToMarshallingStruct(lastInst, structType, structPointer, alloca);
//             sourceStruct = alloca;
//         }
//         else
//         {
//             sourceStruct = structPointer;
//         }

//         BitCastInst *bitcast = new BitCastInst(sourceStruct, PointerType::get(IntegerType::get(context, 8), 0));
//         bitcast->insertAfter(lastInst);
//         lastInst = bitcast;

//         Value *args[2];
//         args[0] = bitcast;
//         args[1] = createInt32Constant(&context, allocSize);

//         Function *setKernelArgHostsideBuffer = cast<Function>(getOrInsertFunction(
//             M,
//             "setKernelArgHostsideBuffer",
//             Type::getVoidTy(context),
//             PointerType::get(IntegerType::get(context, 8), 0),
//             IntegerType::get(context, 32)));

//         CallInst *call = CallInst::Create(setKernelArgHostsideBuffer, ArrayRef<Value *>(args));
//         call->insertAfter(lastInst);
//         lastInst = call;

//         // now we have to handle any pointers, send those through too
//         int i = 0;
//         for (auto pointerit = structInfo->pointerInfos.begin(); pointerit != structInfo->pointerInfos.end(); pointerit++)
//         {
//             PointerInfo *pointerInfo = pointerit->get();
//             vector<Value *> indices;
//             indices.push_back(createInt32Constant(&context, 0));
//             for (auto idxit = pointerInfo->indices.begin(); idxit != pointerInfo->indices.end(); idxit++)
//             {
//                 int idx = *idxit;
//                 indices.push_back(createInt32Constant(&context, idx));
//             }
//             GetElementPtrInst *gep = GetElementPtrInst::CreateInBounds(structType, structPointer, ArrayRef<Value *>(&indices[0], &indices[indices.size()]), "getfloatstaraddr");
//             gep->insertAfter(lastInst);
//             lastInst = gep;

//             LoadInst *loadgep = new LoadInst(gep, "loadgep");
//             loadgep->insertAfter(lastInst);
//             lastInst = loadgep;

//             lastInst = PatchHostside::addSetKernelArgInst_pointer(lastInst, loadgep);
//             i++;
//         }
//         return lastInst;
//     }

    llvm::Instruction *PatchHostside::addSetKernelArgInst(llvm::Instruction *lastInst, ParamInfo *paramInfo)
    {
        /*
        This figures out the type of the argument, and dispatches appropriately. All arguments pass through this function,
        including primitives, pointers, structs, ...
        */
        Value *value = paramInfo->value;
        Value *valueAsPointerInstr = paramInfo->pointer;

        // Indentor indentor_;
        // Indentor indentor;

        Type *devicesideType = paramInfo->typeDevicesideFn;

        // if (paramInfo->devicesideByVal &&
        //     isa<PointerType>(devicesideType) &&
        //     isa<StructType>(cast<PointerType>(devicesideType)->getElementType()))
        // {
        //     // lets walk up the gep???
        //     if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(valueAsPointerInstr))
        //     {
        //         Value *beforeGep = gep->getOperand(0);
        //         valueAsPointerInstr = beforeGep;
        //     }
        //     lastInst = PatchHostside::addSetKernelArgInst_byvaluestruct(lastInst, paramInfo, valueAsPointerInstr);
        // }
        if (IntegerType *intType = dyn_cast<IntegerType>(value->getType()))
        {
            lastInst = PatchHostside::addSetKernelArgInst_int(lastInst, value, intType);
        }
        else if (value->getType()->isFloatingPointTy())
        {
            lastInst = PatchHostside::addSetKernelArgInst_float(lastInst, value);
        }
        else if (value->getType()->isPointerTy())
        {
            Type *elementType = dyn_cast<PointerType>(value->getType())->getElementType();
            // if (isa<StructType>(elementType))
            // {
            //     lastInst = PatchHostside::addSetKernelArgInst_pointerstruct(lastInst, value);
            // }
            // else
            // {
                lastInst = PatchHostside::addSetKernelArgInst_pointer(lastInst, value);
            // }
        }
        // else if (isa<StructType>(value->getType()))
        // {
        //     lastInst = PatchHostside::addSetKernelArgInst_byvaluestruct(lastInst, paramInfo, valueAsPointerInstr);
        // }
        else if (isa<VectorType>(value->getType()))
        {
            // so, this is byval, thus hostside?
            // so send it as a hostside buffer?
            lastInst = PatchHostside::addSetKernelArgInst_byvaluevector(lastInst, valueAsPointerInstr);
        }
        else
        {
            // COCL_LLVM_DUMP(value);
            outs() << "\n";
            // throw std::runtime_error("kernel arg type type not implemented " + typeDumper.dumpType(value->getType()));
            throw std::runtime_error("kernel arg type type not implemented");
        }
        return lastInst;
    }

    void PatchHostside::getLaunchArgValue(GenericCallInst *inst, LaunchCallInfo *info, ParamInfo *paramInfo)
    {
        // input to this is:
        // - inst is cudaSetupArgument instruction, with:
        //   - first operand is a value pointing to the value we want to send to the kernel
        //
        // - output of this method is
        //    populate info with a Value holding the actual concrete value we want to send to the kernel
        //    (note a pointer to it, since we Load the pointer)
        // Notes:
        // - the first operand of inst was created as bitcast(i8*)(alloca (type-of-arg))
        // - the alloca instruction is inst->getOperand(0)->getOperand(0)
        // - so if we load from the alloca instruction, we should have the value we want?
        // outs() << "getLaunchArgValue " << "\n";
        // Indentor indentor;
        int size = (int)cast<ConstantInt>(inst->getOperand(1))->getZExtValue();
        paramInfo->size = size;
        if (!isa<Instruction>(inst->getOperand(0)))
        {
            outs() << "getlaunchvalue, first operatnd of inst is not an instruction..."
                   << "\n";
            // inst->dump();
            // outs() << "\n";
            // COCL_LLVM_DUMP(inst->getOperand(0));
            // outs() << "\n";
            throw runtime_error("getlaunchvalue, first operatnd of inst is not an instruction...");
        }
        Instruction *bitcast = cast<Instruction>(inst->getOperand(0));
        AllocaInst *alloca = 0;
        if (isa<BitCastInst>(bitcast))
        {
            alloca = cast<AllocaInst>(bitcast->getOperand(0));
        }
        else
        {
            alloca = cast<AllocaInst>(bitcast);
        }
        // Type *type = alloca->
        // [Comment from 2022]: so, several years ago, LoadInst didnt need a first argument of Type *
        //                    now it does. I wonder what type it needs? The type of the alloca?
        //                    reminder, llvm code looks a little like:
        //   %4 = alloca i32*, align 8
        //   %5 = alloca i32, align 4
        //   %6 = alloca i32*, align 8
        //   store i32* %0, i32** %4, align 8
        //   store i32 %1, i32* %5, align 4
        //   store i32* %2, i32** %6, align 8
        //   %7 = bitcast i32** %4 to i8*
        //   %8 = call i32 @cudaSetupArgument(i8* %7, i64 8, i64 0)
        //   %9 = icmp eq i32 %8, 0
        //   br i1 %9, label %10, label %20

        // 10:                                               ; preds = %3
        //   %11 = bitcast i32* %5 to i8*
        //   %12 = call i32 @cudaSetupArgument(i8* %11, i64 4, i64 8)
        //   %13 = icmp eq i32 %12, 0
        //   br i1 %13, label %14, label %20

        // 14:                                               ; preds = %10
        //   %15 = bitcast i32** %6 to i8*
        //   %16 = call i32 @cudaSetupArgument(i8* %15, i64 8, i64 16)
        //   %17 = icmp eq i32 %16, 0
        //   br i1 %17, label %18, label %20
        // [/Comment from 2022]
        // std::cout << "dump type alloca get type:" << std::endl;
        // VERIGPU_LLVM_DUMP(alloca->getType());
        Instruction *load = new LoadInst(alloca->getAllocatedType(), alloca, "loadCudaArg", inst->getInst());
        paramInfo->value = load;
        paramInfo->pointer = alloca;
    }

    void PatchHostside::getLaunchTypes(
        llvm::Module *M, const llvm::Module *MDevice, GenericCallInst *inst, LaunchCallInfo *info)
    {
        // input to this is a cudaLaunch instruction
        // sideeffect is to populate in info:
        // - name of the kernel
        // - type of each of the kernel parameters (without the actual Value's)
        // info->callTypes.clear();
        // outs() << "getLaunchTypes()\n";

        // Indentor indentor;

        // the cudaLaunch bitcasts the function to a char *:
        // so we need to walk back along that to get the original function:
        BitCastInst *bitcast = cast<BitCastInst>(cast<ConstantExpr>(inst->getArgOperand(0))->getAsInstruction());
        Function *hostFn = cast<Function>(bitcast->getOperand(0));

        // PointerType *pointerFunctionType = cast<PointerType>(hostFn->getType());
        // FunctionType *hostFnType = cast<FunctionType>(pointerFunctionType->getElementType());

        info->stubKernelName = hostFn->getName().str();
        // info->kernelName = info->stubKernelName;
        // size_t pos = info->kernelName.find("__device_stub__", 0);
        // info->kernelName.replace(pos, strlen("__device_stub__"), "_");
        info->kernelName = nameRemoveDeviceStub(info->stubKernelName);

        // Function *deviceFn = MDevice->getFunction(info->kernelName);
        Function *deviceFn = M->getFunction(info->stubKernelName);
        if (deviceFn == 0)
        {
            cout << "ERROR: failed to find device kernel [" << info->kernelName << "]" << endl;
            throw runtime_error("ERROR: failed to find device kernel " + info->kernelName);
        }
        FunctionType *deviceFnType = cast<FunctionType>(cast<PointerType>(deviceFn->getType())->getElementType());

        int i = 0;
        vector<Argument *> devicesideArgs;
        for (auto it = deviceFn->arg_begin(); it != deviceFn->arg_end(); it++)
        {
            Argument *arg = &*it;
            devicesideArgs.push_back(arg);
        }
        for (auto it = deviceFnType->param_begin(); it != deviceFnType->param_end(); it++)
        {
            Type *typeDevicesideFn = *it;
            if (i >= (int)info->params.size())
            {
                cout << "warning: exceeded number of params" << endl;
                break;
            }
            {
                // Indentor indentor;
                ParamInfo *paramInfo = info->params[i].get();

                paramInfo->paramIndex = i;

                paramInfo->devicesideArg = devicesideArgs[i];
                paramInfo->devicesideByVal = paramInfo->devicesideArg->hasByValAttr();

                paramInfo->typeDevicesideFn = typeDevicesideFn;
            }
            i++;
        }
    }

    void PatchHostside::patchCudaLaunch(
        llvm::Module *M, const llvm::Module *MDevice, llvm::Function *F,
        GenericCallInst *inst, std::vector<llvm::Instruction *> &to_replace_with_zero)
    {
        // This replaces the call to cudaLaunch with calls to setup the arguments, then
        // a call to kernelGo
        // We pass the informatino about the kernel first, then the arguments, then finally
        // call kernelGo.

        // MDevice is just so we can see how the device-side kernel declarations look, we dont
        // modify/patch it in any way here

        PatchHostside::getLaunchTypes(M, MDevice, inst, launchCallInfo.get());
        to_replace_with_zero.push_back(inst->getInst());

        string kernelName = launchCallInfo->kernelName;
        Instruction *kernelNameValue = addStringInstr(M, "s_" + ::deviceriscvcode_stringname + "_" + kernelName, kernelName);
        kernelNameValue->insertBefore(inst->getInst());

        // this isnt actually needed for running, but hopefully useful for debugging
        Instruction *llSourcecodeValue = addStringInstrExistingGlobal(M, deviceriscvcode_stringname);
        llSourcecodeValue->insertBefore(inst->getInst());

        Function *configureKernel = cast<Function>(getOrInsertFunction(
            F->getParent(),
            "configureKernel",
            Type::getVoidTy(context),
            PointerType::get(IntegerType::get(context, 8), 0),
            PointerType::get(IntegerType::get(context, 8), 0)
            // PointerType::get(IntegerType::get(context, 8), 0),
            ));

        Value *args[] = {kernelNameValue, llSourcecodeValue};
        CallInst *callConfigureKernel = CallInst::Create(configureKernel, ArrayRef<Value *>(&args[0], &args[2]));
        callConfigureKernel->insertBefore(inst->getInst());
        Instruction *lastInst = callConfigureKernel;

        // pass args now
        int i = 0;
        for (auto argit = launchCallInfo->params.begin(); argit != launchCallInfo->params.end(); argit++)
        {
            ParamInfo *paramInfo = argit->get();
            lastInst = PatchHostside::addSetKernelArgInst(lastInst, paramInfo);
            i++;
        }
        // trigger the kernel...
        Function *kernelGo = cast<Function>(F->getParent()->getOrInsertFunction(
            "kernelGo",
            Type::getVoidTy(context),
            static_cast<size_t>(NULL)).getCallee());
        CallInst *kernelGoInst = CallInst::Create(kernelGo);
        kernelGoInst->insertAfter(lastInst);
        lastInst = kernelGoInst;

        launchCallInfo->params.clear();
    }

    void PatchHostside::patchFunction(llvm::Module *M, const llvm::Module *MDevice, llvm::Function *F)
    {
        // this will take the calls to cudaSetupArgument(someArg, argSize, ...), and
        // cudaLaunch(function), and rewrite them to call our Gpu runtime instead
        // we do a bunch of bytecode parsing, to figure out how exactly we are goign to get the arguments
        // into coriander
        // For example, if it's a by-value struct, we're going to have to do some hacking
        // If the by-value struct contains pointers, it'll need a bit (lot :-P) more hacking

        // MDevice is only for information, so we can see the declaration of kernels on the device-side

        vector<Instruction *> to_replace_with_zero;
        IntegerType *inttype = IntegerType::get(context, 32);
        ConstantInt *constzero = ConstantInt::getSigned(inttype, 0);
        launchCallInfo->params.clear();
        // Indentor indentor;
        for (auto it = F->begin(); it != F->end(); it++)
        {
            BasicBlock *basicBlock = &*it;
            for (auto insit = basicBlock->begin(); insit != basicBlock->end(); insit++)
            {
                Instruction *inst = &*insit;
                if (!isa<CallInst>(inst) && !isa<InvokeInst>(inst))
                {
                    continue;
                }
                Function *called = 0;
                unique_ptr<GenericCallInst> genCallInst;
                if (CallInst *callInst = dyn_cast<CallInst>(inst))
                {
                    called = callInst->getCalledFunction();
                    genCallInst = GenericCallInst::create(callInst);
                }
                else if (InvokeInst *callInst = dyn_cast<InvokeInst>(inst))
                {
                    called = callInst->getCalledFunction();
                    genCallInst = GenericCallInst::create(callInst);
                }
                if (called == 0)
                {
                    continue;
                }
                if (!called->hasName())
                {
                    continue;
                }
                string calledFunctionName = called->getName().str();
                if (calledFunctionName == "cudaSetupArgument")
                {
                    unique_ptr<ParamInfo> paramInfo(new ParamInfo());
                    PatchHostside::getLaunchArgValue(genCallInst.get(), launchCallInfo.get(), paramInfo.get());
                    launchCallInfo->params.push_back(std::move(paramInfo));
                    to_replace_with_zero.push_back(inst);
                }
                else if (calledFunctionName == "cudaLaunch")
                {
                    PatchHostside::patchCudaLaunch(M, MDevice, F, genCallInst.get(), to_replace_with_zero);
                }
            }
        }
        for (auto it = to_replace_with_zero.begin(); it != to_replace_with_zero.end(); it++)
        {
            Instruction *inst = *it;
            BasicBlock::iterator ii(inst);
            if (InvokeInst *invoke = dyn_cast<InvokeInst>(inst))
            {
                // need to add an uncondtioinal branch, after the old invoke locaiton
                BasicBlock *oldTarget = invoke->getNormalDest();
                BranchInst *branch = BranchInst::Create(oldTarget);
                branch->insertAfter(inst);
            }
            ReplaceInstWithValue(inst->getParent()->getInstList(), ii, constzero);
        }
    }

    std::string PatchHostside::getBasename(std::string path)
    {
        // grab anything after final / ,or whole string
        size_t slash_pos = path.rfind('/');
        if (slash_pos == string::npos)
        {
            return path;
        }
        return path.substr(slash_pos + 1);
    }

    void PatchHostside::patchModule(Module *M, const Module *MDevice)
    {
        // entry point: given Module M, traverse all functions, rewriting the launch instructison to call
        // into Coriander runtime

        // MDevice is only for information, so we can see the declaration of kernels on the device-side

        ifstream f_inll(::deviceriscvfilename);
        string deviceriscv_sourcecode(
            (std::istreambuf_iterator<char>(f_inll)),
            (std::istreambuf_iterator<char>()));

        ::deviceriscvcode_stringname = "__deviceriscv_sourcecode" + ::deviceriscvfilename;
        addGlobalVariable(M, deviceriscvcode_stringname, deviceriscv_sourcecode);

        for (auto it = M->begin(); it != M->end(); it++)
        {
            Function *F = &*it;
            PatchHostside::patchFunction(M, MDevice, F);
            verifyFunction(*F);
        }
    }

} // namespace cocl

int main(int argc, char *argv[])
{
    SMDiagnostic smDiagnostic;
    argparsecpp::ArgumentParser parser;

    string rawhostfilename;
    string patchedhostfilename;


    parser.add_string_argument("--hostrawfile", &rawhostfilename)->required()->help("input file");
    parser.add_string_argument("--devicellfile", &::devicellfilename)->required()->help("input file");
    parser.add_string_argument("--deviceriscvfile", &::deviceriscvfilename)->required()->help("input file");
    parser.add_string_argument("--hostpatchedfile", &patchedhostfilename)->required()->help("output file");
    if (!parser.parse_args(argc, argv))
    {
        return -1;
    }

    std::unique_ptr<llvm::Module> module = parseIRFile(rawhostfilename, smDiagnostic, context);
    if (!module)
    {
        smDiagnostic.print(argv[0], errs());
        return 1;
    }
    std::unique_ptr<llvm::Module> deviceModule = parseIRFile(devicellfilename, smDiagnostic, context);
    if (!deviceModule)
    {
        smDiagnostic.print(argv[0], errs());
        return 1;
    }

    try
    {
        PatchHostside::patchModule(module.get(), deviceModule.get());
    }
    catch (const runtime_error &e)
    {
        cout << endl;
        cout << "Something went wrong, sorry." << endl;
        cout << endl;
        cout << "More detail for devs/maintainers:" << endl;
        cout << "  exception: " << e.what() << endl;
        cout << "  rawhost ll file: " << rawhostfilename << "\n";
        outs() << "  hostpatched file: " << patchedhostfilename << "\n";
        cout << endl;
        return -1;
    }

    AssemblyAnnotationWriter assemblyAnnotationWriter;
    ofstream ofile;
    ofile.open(patchedhostfilename);
    raw_os_ostream my_raw_os_ostream(ofile);
    verifyModule(*module);
    module->print(my_raw_os_ostream, &assemblyAnnotationWriter);
    ofile.close();
    return 0;
}
