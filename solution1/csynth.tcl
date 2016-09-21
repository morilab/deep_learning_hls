############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2016 Xilinx, Inc. All Rights Reserved.
############################################################
open_project deep_learning_hls
set_top func_01
add_files deep_learning_hls/Matrix.h
add_files deep_learning_hls/convolution_perceptron.h
add_files deep_learning_hls/deep_learning.cpp
add_files deep_learning_hls/deep_learning.h
add_files deep_learning_hls/perceptron_fnn.h
add_files deep_learning_hls/relu_perceptron_fnn.h
add_files deep_learning_hls/softmax_perceptron_fnn.h
add_files -tb deep_learning_hls/MNIST.cpp
add_files -tb deep_learning_hls/MNIST.h
add_files -tb deep_learning_hls/deep_learning.cpp
add_files -tb deep_learning_hls/deep_learning.h
add_files -tb deep_learning_hls/test.cpp
add_files -tb deep_learning_hls/test.h
open_solution "solution1"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10ns -name default
#source "./deep_learning_hls/solution1/directives.tcl"
csynth_design