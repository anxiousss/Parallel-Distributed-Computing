cmake_minimum_required(VERSION 3.28)
project(Parallel_Distributed_Computing CUDA)
enable_language(C)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80 86)

add_executable(hw1 "hw1/main.cu")
add_executable(hw2 "hw2/main.cu")
add_executable(lab1 "lab1/main.cu")
add_executable(lab2 "lab2/main.cu")
add_executable(lab3 "lab3/main.cu")
add_executable(lab4 "lab4/main.cu")