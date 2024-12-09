include(CMakeParseArguments)

function(add_cuda_executable target)
    set(prefix ARG)
    set(noValues "")
    set(singleValues "SRC")
    set(multiValues "LINK_PUBLIC" "LINK_PRIVATE")
    cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
        "${multiValues}" ${ARGN})
    
    add_executable(${target} ${ARG_SRC})
    target_link_libraries(${target} 
    PUBLIC
    CUDA::cudart 
    ${ARG_LINK_PUBLIC}
    PRIVATE 
    ${ARG_LINK_PRIVATE})
    target_include_directories(${target}
    PUBLIC
    ${HELLO_CUDA_ROOT_DIR})
endfunction()

function(add_cuda_shared_library target)
    set(prefix ARG)
    set(noValues "")
    set(singleValues "SRC")
    set(multiValues "LINK_PUBLIC" "LINK_PRIVATE")
    cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
        "${multiValues}" ${ARGN})
    
    add_library(${target} SHARED ${ARG_SRC})
    target_link_libraries(${target} 
    PUBLIC
    CUDA::cudart 
    ${ARG_LINK_PUBLIC}
    PRIVATE 
    ${ARG_LINK_PRIVATE})
    target_include_directories(${target}
    PUBLIC
    ${HELLO_CUDA_ROOT_DIR})
endfunction()

function(add_gtest target)
    set(prefix ARG)
    set(noValues "")
    set(singleValues "SRC")
    set(multiValues "LINK_PUBLIC" "LINK_PRIVATE")
    cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
        "${multiValues}" ${ARGN})
    
    add_executable(${target} ${ARG_SRC})
    target_link_libraries(${target} 
    PUBLIC
    ${ARG_LINK_PUBLIC}
    Utils
    PRIVATE
    GTest::gtest
    GTest::gtest_main 
    ${ARG_LINK_PRIVATE})
    target_include_directories(${target}
    PUBLIC
    ${HELLO_CUDA_ROOT_DIR})

    add_test(NAME ${target} COMMAND ${target})
endfunction()