add_cuda_executable(PointwiseAddNaive 
SRC 
PointwiseAddNaive.cu
LINK_PUBLIC 
Utils
)

add_cuda_executable(PointwiseAddTiled
SRC 
PointwiseAddTiled.cu
LINK_PUBLIC 
Utils
)