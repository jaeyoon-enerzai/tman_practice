# customized QNN execution repository

## Step1 - make a graph only with Add op for float tensor

- Done
- see 0d0e2131 commit
- Note : You should not use non-defined SOC enum. Our target device was QCS6490 but there was no enum for it. I saw in executorch, there is a enum for QCS6490 and thought that I might be able to apply that enum number. But, it seems not. If one does like that, it seems taht SOC number is not recognized and goes fallback to some default SOC which lead to not supporting FP16 error.

## Step2 - add one additional op on the graph
- Which type of tensor should we use it? 
  - APP READ/WRITE for graph input/output
  - NATIVE for intermediate tensor

## Step3 - add an op which needs static parameter into the graph
- z = x+y & out = x + c(static)

## Step4 - add matmul op into the graph. One input must be an intermediate tensor
- z1 = x+y & z2 = x + c (static) & out = z1 * z2

## Step5 - add one transformer block with weight being transformed. without kv-cache

## Step6 - execute the graph and check the result value

## Step7 - Add a shared buffer for kv cache - see MemoryManager

## Step8 - Add a profiler

## Step9 - Add PreRegister

## Step10 - Add a quantization
- Blockwise config

## Step11 - Add a customized op package
- see QnnBackendCommon.cpp

## Step12 - Visualize the graph

## Step13 - Prefill/Decoding separation (Multi-graph?? Multi method??)