add_executable(HelloCuda HelloCuda.cpp)
target_link_libraries(HelloCuda 
PUBLIC 
CUDA::cudart)
set_target_properties(HelloCuda PROPERTIES CUDA_ARCHITECTURES "89")
set_source_files_properties(HelloCuda.cpp PROPERTIES LANGUAGE CUDA)