# Hello CUDA (CUDA Tutorials)

My notes and code examples studied while understanding CUDA.

## Dependenices

- CUDA Toolkit
- Clang
- Ninja
- GTest

## Build

This repo uses cmake and ninja. A helper makefile shim is provided to make build, rebuild and
testing easy.

| NOTE |
|------|
| By default, makefile shim uses CUDA 12.3 and compute capability of 89. |

### Build
```bash
git clone https://github.com/shelkesagar29/HelloCUDA.git
cd HelloCUDA
make all 
```

Built code examples can be found at
`build/src`.

### Clean & Rebuild
```bash
make reconfigure && make build
```

## Test

This repo uses `ctest` to run unit tests. To run all tests, run the following command.

```bash
make test
```

Tests are written for utility functions and each kernel. Both functional and benchmark kernel tests
are written.

## Lessons
- **DeviceInfo**  
Provides detailed device info.
- **GEMM**  
GEMM implementations. From Naive to optimized.
- **HelloCdua**  
Hello world from CUDA
- **Pointise**  
Pointwise addition example.  

- **Utils**  
Some useful utilities.