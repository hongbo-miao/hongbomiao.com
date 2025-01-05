#include <iostream>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

int main() {
  llvm::LLVMContext Context;
  llvm::Module *ModuleObj = new llvm::Module("add_module", Context);
  llvm::IRBuilder<> Builder(Context);

  // Create add function
  std::vector<llvm::Type *> Ints(2, llvm::Type::getInt32Ty(Context));
  llvm::FunctionType *FT = llvm::FunctionType::get(llvm::Type::getInt32Ty(Context), Ints, false);
  llvm::Function *F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, "add", ModuleObj);

  // Create basic block
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Context, "entry", F);
  Builder.SetInsertPoint(BB);

  // Get function parameters
  llvm::Function::arg_iterator args = F->arg_begin();
  llvm::Value *x = args++;
  llvm::Value *y = args++;

  // Create add instruction
  llvm::Value *result = Builder.CreateAdd(x, y, "result");
  Builder.CreateRet(result);

  // Create execution engine
  llvm::ExecutionEngine *EE =
    llvm::EngineBuilder(std::unique_ptr<llvm::Module>(ModuleObj)).create();

  // Run function
  std::vector<llvm::GenericValue> Args(2);
  Args[0].IntVal = llvm::APInt(32, 10);
  Args[1].IntVal = llvm::APInt(32, 32);
  llvm::GenericValue GV = EE->runFunction(F, Args);

  std::cout << GV.IntVal.getSExtValue() << std::endl;

  delete EE;
  return 0;
}
