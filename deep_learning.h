#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ap_int.h>
//#include <hls_math.h>
#include "Matrix.h"
#include "perceptron_fnn.h"
#include "relu_perceptron_fnn.h"
#include "softmax_perceptron_fnn.h"
#include "convolution_perceptron.h"

template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, const int SIZE1, const int SIZE2, const int SIZE3, const int SIZE4, typename T>
Matrix<1,SIZE4,T> convolution_nn(
		Matrix<IN_X,IN_Y,T> inframe,
		T L1_filter[SIZE1][CNV_X*CNV_Y],
		T L1_bias  [SIZE1],
		T L2_filter[SIZE2][SIZE1][CNV_X*CNV_Y],
		T L2_bias  [SIZE2][SIZE1],
		T L3_weight[SIZE3][SIZE2][IN_X/4*IN_Y/4],
		T L3_bias  [SIZE3],
		T L4_weight[SIZE4][SIZE3],
		T L4_bias  [SIZE4]
);

//-------------------------------------------------------------------
// wrapper for VivadoHLS
//-------------------------------------------------------------------
typedef ap_fixed  <8, 2, AP_RND, AP_SAT> raw_internal_t;
typedef ap_ufixed <8, 8, AP_RND, AP_SAT> u8_t;
struct func_01_result_t {
	raw_internal_t is_0;
	raw_internal_t is_1;
	raw_internal_t is_2;
	raw_internal_t is_3;
	raw_internal_t is_4;
	raw_internal_t is_5;
	raw_internal_t is_6;
	raw_internal_t is_7;
	raw_internal_t is_8;
	raw_internal_t is_9;
};
const int SIZE1 = 4;//20;
const int SIZE2 = 4;//20;
const int SIZE3 = 32;//500;
const int SIZE4 = 10;
const int WINDOW_SIZE = 5*5;

func_01_result_t func_01(
		u8_t  in[28*28],
		raw_internal_t L1_filter[SIZE1][WINDOW_SIZE],        // 20x(5x5)=500
		raw_internal_t L1_bias  [SIZE1],                     // =20
		raw_internal_t L2_filter[SIZE2][SIZE1][WINDOW_SIZE], // 20x20x(5x5)=10,000
		raw_internal_t L2_bias  [SIZE2][SIZE1],              // 20x20=400
		raw_internal_t L3_weight[SIZE3][SIZE2][7*7],         // 500x20x(7x7)=490,000
		raw_internal_t L3_bias  [SIZE3],                     // =500
		raw_internal_t L4_weight[SIZE4][SIZE3],              // 10x500=1,000
		raw_internal_t L4_bias  [SIZE4]                      // =10
);

void Layer1(
	u8_t  in[28*28],
	raw_internal_t L1_filter[SIZE1][WINDOW_SIZE],
	raw_internal_t L1_bias  [SIZE1],
	raw_internal_t L1_out   [SIZE1][14*14]
);

void Layer2(
	raw_internal_t in     [SIZE1][14*14],
	raw_internal_t filter [SIZE2][SIZE1][WINDOW_SIZE],
	raw_internal_t bias   [SIZE2][SIZE1],
	raw_internal_t out    [SIZE2][7*7]
);

void Layer3(
	raw_internal_t in     [SIZE2][7*7],
	raw_internal_t filter [SIZE3][SIZE2][7*7],
	raw_internal_t bias   [SIZE3],
	raw_internal_t out    [SIZE3]
);

void Layer4(
	raw_internal_t in     [SIZE3],
	raw_internal_t filter [SIZE4][SIZE3],
	raw_internal_t biasa  [SIZE4],
	raw_internal_t out    [SIZE4]
);
//-------------------------------------------------------------------
// wrapper for VivadoHLS
//-------------------------------------------------------------------
template <const int X, const int Y, typename T>
Matrix<1,Y,T> deep_learning(T x[X], T weight[Y][X], T bias[Y]) {
	relu_perceptron_fnn<X,Y,T> fnn;

	DeepLearning_Init_L1 :
	for(int i=0;i<Y;i++){
		fnn.bias(0,i)=bias[i];
		DeepLearning_Init_L2 :
		for(int j=0;j<X;j++){
			fnn.in(0,j)=x[j];
			fnn.weight(j,i)=weight[i][j];
		}
	}
	fnn.run();
	return fnn.out;
}
