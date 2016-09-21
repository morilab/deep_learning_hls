############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2016 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_array_reshape -type cyclic -factor 25 -dim 2 "func_01" L1_filter
set_directive_array_reshape -type cyclic -factor 25 -dim 3 "func_01" L2_filter
set_directive_dataflow "func_01"
set_directive_array_partition -type complete -dim 1 "func_01" inframe
set_directive_pipeline -II 1 "convolution_nn"
