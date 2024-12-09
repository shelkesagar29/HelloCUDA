## Pointwise Ops

Pointwise ops are one of the simplest in deep learning. These are element-wise operations applied to the individual elements of tensor.

This folder implements two kernels for pointwise addition op.

### Naive

Each thread access global memory directly.

### Tiled

Tiled kernel implements addition using tiling.

NOTE: In this case, tiling has no effect on perf since we are effectively accessing global memory
only once in case of both naive and tiled implementation.

