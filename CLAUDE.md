# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Customized QNN (Qualcomm Neural Network) SDK project for running quantized transformer inference on Qualcomm Hexagon NPU (HTP backend). The project builds QNN compute graphs ahead-of-time on an x86 host, serializes them to a context binary, then executes on-device (Android aarch64). The target device is **QCS6490** with **HTP v73** architecture.

Custom HVX operations (`TMANOpPackage`) implement 4-bit quantized matrix multiplication with group_size=128, symmetric quantization. The custom ops are: `TMANPrecompute`, `TMANLinear`, `TMANFinalize`.

## Build Commands

QNN SDK is at `/workspace/2.40.0.251030/`. Android NDK at `/workspace/android-ndk-r26c/`.

### AOT compiler (x86 host, builds graph → serializes to .bin)
```bash
cd /workspace/compile/script && bash build.sh aot
```
This creates `build_aot/aot/qnn_offline_compiler`.

### Runtime runner (Android aarch64, loads .bin → executes on device)
```bash
cd /workspace/compile/script && bash build.sh runtime
```
This creates `build_runtime/runtime/qnn_runtime_runner`.

### Run the AOT compiler (with custom op package)
```bash
cd /workspace/compile/script && bash compile.sh
```
This sets `QNN_OP_PACKAGE_PATHS` to load `TMANOpPackage` and runs the offline compiler.

### Deploy and run on device
```bash
cd /workspace/compile/script && bash run.sh
```
Pushes binary, weights, and runner to `/data/local/tmp/htprun` via adb, then executes.

### Profiling
```bash
adb pull /data/local/tmp/htprun/qnn.log output/qnn-profiling-data_0.log
/workspace/2.40.0.251030/bin/x86_64-linux-clang/qnn-profile-viewer \
  --config config.json \
  --reader /workspace/2.40.0.251030/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so \
  --input_log output/qnn-profiling-data_0.log \
  --schematic kv_forward_schematic.bin \
  --output ./chrometrace.json
```

### Building the custom op package (TMANOpPackage)
Located at `/workspace/TMANOpPackage/` (external to this repo). Requires Hexagon SDK and QNN SDK sourced:
```bash
cd ${HEXAGON_SDK_ROOT} && source setup_sdk_env.source
cd ${QNN_SDK_ROOT}/bin && source envsetup.sh
cd /workspace/TMANOpPackage && make htp_x86 htp_v73
```

## Architecture

### Two-phase execution model

**Phase 1 — AOT (`aot/src/main_aot.cpp`):** Runs on x86 host. Constructs QNN graphs by defining tensors, registering them, creating ops (via `OpHolder` + `MakeOpHolder`), validating, and adding nodes. Finalizes the graph and serializes the context to `multi_graph.bin`. Static weights (quantized) are loaded from external `.bin` files and embedded into the graph.

**Phase 2 — Runtime (`runtime/src/main_run.cpp`):** Runs on Android device. Loads `multi_graph.bin`, recreates graphs from binary via `CreateFromBinary`, allocates shared buffers (RPC memory for DSP), registers tensors in a shared arena, fills inputs, calls `graphExecute`, and dumps/validates outputs. Includes CPU reference matmul for verification.

### Common library (`common/`)

Static library `qnn_common` wrapping QNN SDK C APIs into C++ RAII classes. Lifecycle follows a strict dependency chain:

```
QnnDynLoad (singleton, loads .so)
  → QnnBackendRuntime (backend + op package registration)
    → QnnDeviceRuntime (device config: SOC, VTCM, arch)
      → QnnContextRuntime (context, supports weight sharing + multi-context)
        → QnnGraphRuntime (graph creation/restore, tensor registration, finalize)
```

Supporting classes:
- **QnnTensor** — wraps `Qnn_Tensor_t` v2, manages dims/name/data ownership. Tensor types: `APP_WRITE` (input), `APP_READ` (output), `NATIVE` (intermediate), `STATIC` (weights).
- **QnnProfilerRuntime** — profiling with optrace level, serialization to log files
- **SharedBuffer** — singleton wrapping `libcdsprpc.so` RPC memory (AllocMem/MemToFd/FreeMem) with arena-based sub-allocation
- **QnnMemManagerRuntime** — registers tensors into shared buffer arenas for zero-copy DSP access
- **HtpBackendCacheRuntime** — parses context binary to extract graph I/O metadata and spill-fill buffer sizes

### Key configuration (hardcoded in `qnn_device.h`)

- SOC: `QCS6490` (enum 93)
- HTP Arch: `V73`
- VTCM: 2 MB
- PD Session: Unsigned
- Graph: VTCM 8MB, opt level 3.0, DLBC enabled

### Quantization parameters (in `main_aot.cpp`)

- `GROUP_SIZE = 128`, `BITS = 4`, `SYMMETRIC = 1`
- Weight repacking files: `w_repacked.bin`, `s_repacked.bin` from `/workspace/m2048_k8192_g128/`

### KV-cache (decoding) graph pipeline

```
x(fp32) → Reshape → Cast(fp32→fp16) → TMANPrecompute → [TMANLinear → TMANFinalize → Reshape → Cast(fp16→fp32)]
```
Operations in brackets are currently commented out — being incrementally enabled.

## Important Notes

- **SOC enum pitfall**: Do not use undefined SOC enums. Even if an enum exists (e.g., in executorch), QNN SDK may not recognize it, causing fallback to a default SOC that doesn't support FP16.
- **QnnDynLoad is a keep-alive singleton**: This works around a QNN SDK bug where freeing the backend doesn't clear custom op memory properties (`DEF_TENSOR_PROPERTIES`), causing errors on reload.
- **Tensor version**: All tensors use v2 (`QNN_TENSOR_VERSION_2`). Access fields via `QNN_TENSOR_VER_PTR(t)` macro.
- **Op config version**: Uses v1. Access via `QNN_OP_VER_PTR(cfg)` macro.
- **AOT uses clang++ with libc++** (`-stdlib=libc++`). Runtime uses NDK with `c++_static`.
- **`OpHolder`** stores name, inputs, outputs, params by value. Call `bind()` after populating to wire up the `Qnn_OpConfig_t` pointers. The `MakeOpHolder` template takes a lambda for adding scalar params.
