# customized QNN execution repository

## Step1 - make a graph only with Add op for float tensor

- Done
- see 0d0e2131 commit
- Note : You should not use non-defined SOC enum. Our target device was QCS6490 but there was no enum for it. I saw in executorch, there is a enum for QCS6490 and thought that I might be able to apply that enum number. But, it seems not. If one does like that, it seems taht SOC number is not recognized and goes fallback to some default SOC which lead to not supporting FP16 error.

## Step2 - add one additional op on the graph
- Which type of tensor should we use it? APP READ/WRITE? or NATIVE?

## Step3 - add an op which needs static parameter into the graph

## Step4 - add matmul op into the graph. One input must be an intermediate tensor

## Step5 - add memory manager

## Step6 - add one transformer block with weight being transformed. add kv-cache

## Step7 - execute the graph and check the result value

## Step8 - Add a profiler

## Step9 - Add a quantization
