#include "deep_learning.h"

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
) {
	convolution_perceptron<IN_X  ,IN_Y  ,CNV_X,CNV_Y,T> L1_perceptron[SIZE1];
	convolution_perceptron<IN_X/2,IN_Y/2,CNV_X,CNV_Y,T> L2_perceptron[SIZE2];
	relu_perceptron_fnn<(IN_X/4*IN_Y/4)*SIZE2,SIZE3,T> L3_connect;
	softmax_perceptron_fnn<SIZE3,SIZE4,T> L4_out;

	// Layer 1 (Input Layer)
	for(int i=0;i<SIZE1;i++){
		L1_perceptron[i].in = inframe;
		L1_perceptron[i].set_bias(L1_bias[i]);
		L1_perceptron[i].set_filter(L1_filter[i]);
		L1_perceptron[i].Proc();
	}
	printf("L_perceptron[0]:\n");
	L1_perceptron[0].view_filter();
	L1_perceptron[0].view_conv();
	L1_perceptron[0].view_out();

	// Layer 2
	for(int j=0;j<SIZE2;j++){
		L2_perceptron[j].clear_conv();
		for(int i=0;i<SIZE1;i++){
			L2_perceptron[j].in = L1_perceptron[i].out;
			L2_perceptron[j].set_bias(L2_bias[j][i]);
			L2_perceptron[j].set_filter(L2_filter[j][i]);
			L2_perceptron[j].Convolution();
		}
		L2_perceptron[j].Activation();
		L2_perceptron[j].MaxPooling();

		for(int y=0;y<IN_Y/4;y++){
			for(int x=0;x<IN_X/4;x++){
				const int index  = (IN_X/4)*y+x;
				const int offset = (IN_X/4*IN_Y/4)*j;
				L3_connect.in(0,offset+index) = L2_perceptron[j].out(x,y);
			}
		}
	}

	// Layer 3
	for(int k=0;k<SIZE3;k++){
		L3_connect.bias(0,k)=L3_bias[k];
		for(int j=0;j<SIZE2;j++){
			for(int i=0;i<IN_X/4*IN_Y/4;i++){
				L3_connect.weight(j*(IN_X/4*IN_Y/4)+i,k) = L3_weight[k][j][i];
			}
		}
	}
	L3_connect.run();
	//L3_connect.out.view_float("L3.out");
	for(int i=0;i<SIZE3;i++){
		L4_out.in(0,i) = L3_connect.out(0,i);
	}

	// Layer 4 (Output Layer)
	for(int j=0;j<SIZE4;j++){
		L4_out.bias(0,j)=L4_bias[j];
		for(int i=0;i<SIZE3;i++){
			L4_out.weight(i,j) = L4_weight[j][i];
		}
	}
	L4_out.run();
	L4_out.out.view_float("L4.out");

	return L4_out.out;
}

//-------------------------------------------------------------------
// wrapper for VivadoHLS
//-------------------------------------------------------------------
func_01_result_t func_01(
	u8_t  in[28*28],
	raw_internal_t L1_filter[SIZE1][WINDOW_SIZE],        // 20x(5x5)=500
	raw_internal_t L1_bias  [SIZE1],                     // =20
	raw_internal_t L2_filter[SIZE2][SIZE1][WINDOW_SIZE], // 20x20x(5x5)=10,000
	raw_internal_t L2_bias  [SIZE2][SIZE1],              // 20x20=400
	raw_internal_t L3_weight[SIZE3][SIZE2][7*7],         // 500x20x(7x7)=490,000
	raw_internal_t L3_bias  [SIZE3],                     // =500
	raw_internal_t L4_weight[SIZE4][SIZE3],              // 10x500=1,000
	raw_internal_t L4_bias  [SIZE4])                     // =10
{
	Matrix<28,28,raw_internal_t> inframe;
	Matrix<1,SIZE4,raw_internal_t> result;

	for(int y=0;y<28;y++){
		for(int x=0;x<28;x++){
			inframe(x,y) = in[y*28+x];
		}
	}

	result = convolution_nn<28,28 ,5,5 ,SIZE1,SIZE2,SIZE3,SIZE4 ,raw_internal_t>
	         (inframe, L1_filter, L1_bias, L2_filter, L2_bias ,L3_weight ,L3_bias ,L4_weight ,L4_bias);

	func_01_result_t func_01;
	func_01.is_0 = result(0,0);
	func_01.is_1 = result(0,1);
	func_01.is_2 = result(0,2);
	func_01.is_3 = result(0,3);
	func_01.is_4 = result(0,4);
	func_01.is_5 = result(0,5);
	func_01.is_6 = result(0,6);
	func_01.is_7 = result(0,7);
	func_01.is_8 = result(0,8);
	func_01.is_9 = result(0,9);

	return func_01;
}

void Layer4(
	raw_internal_t in     [SIZE3],
	raw_internal_t weight [SIZE4][SIZE3],
	raw_internal_t bias   [SIZE4],
	raw_internal_t out    [SIZE4])
{
	softmax_perceptron_fnn<SIZE3,SIZE4,raw_internal_t> fnn;

	Layer4_in_L1 :
	for(int i=0;i<SIZE3;i++){
		fnn.in(0,i) = in[i];
	}

	Layer4_Param_L1 :
	for(int j=0;j<SIZE4;j++){
		fnn.bias(0,j)=bias[j];
		Layer4_Param_L2 :
		for(int i=0;i<SIZE3;i++){
			fnn.weight(i,j) = weight[j][i];
		}
	}
	fnn.run();

	Layer4_out_L1 :
	for(int i=0;i<SIZE4;i++){
		out[i] = fnn.out(0,i);
	}
}
