#include "test.h"

int main(int argc, char** argv){
	string imgfile = argv[1];
	string lblfile = argv[1];

	imgfile += "train-images.idx3-ubyte";
	lblfile += "train-labels.idx1-ubyte";
	printf("Image File = [%s]\n",imgfile.c_str());
	printf("Label File = [%s]\n",lblfile.c_str());

	if(!mnist.read_images(imgfile.c_str())){
		printf("Can't read Image File.\n");
		return 1;
	}
	if(!mnist.read_labels(lblfile.c_str())){
		printf("Can't read Label File.\n");
		return 1;
	}

	//test_01(); // deep_learning関数による4入力3出力ニューラルネット試験
	//test_02(); // 畳み込みパーセプトロン単体試験
	//test_03(); // ディープラーニング試験(1)
	test_04(); // MNISTファイル読み込み試験
	//test_05(); // ディープラーニング試験(2)

	return 0;
}

//-------------------------------------------------------------------------------------
// deep_learning関数による4入力3出力ニューラルネット動作
//-------------------------------------------------------------------------------------
int test_01(void){
	//------------------------------------
	// setting input data (x)
	//------------------------------------
	int x[4] =
		{1,3,5,7};

	//------------------------------------
	// setting Parameter (bias,weight)
	//------------------------------------
	int bias[3] =
		{-1,0,1};

	int weight[3][4] = {
		{ 2, 2, 2, 2},
		{ 2, 2,-8,-4},
		{ 2,-1,-2, 2},
	};

	//------------------------------------
	// View Input data & Parameter
	//------------------------------------
	for(int i=0;i<4;i++){
		printf("x%d = %d\n",i,x[i]);
	}
	printf("\n");

	printf("Weight = {\n");
	for(int i=0;i<4;i++){
		printf("  {");
		for(int j=0;j<3;j++){
			printf(" %3d",weight[j][i]);
		}
		printf(" }\n");
	}
	printf("};\n\n");

	printf("Bias = {\n  {");
	for(int i=0;i<3;i++){
		printf(" %3d",bias[i]);
	}
	printf(" }\n};\n\n");

	//------------------------------------
	// run DeepLearning
	//------------------------------------
	Matrix<1,3,int> result;
	result = deep_learning<4,3,int>(x, weight, bias);

	for(int i=0;i<3;i++){
		printf("u%d = %d\n",i,result(0,i));
	}

	return 0;
}

//-------------------------------------------------------------------------------------
// 畳み込みパーセプトロン単体試験
//-------------------------------------------------------------------------------------
int test_02(void){
	Matrix<28,28,int> inframe;
	convolution_perceptron<28,28,5,5,int> tr;

	// inframe初期化
	for(int y=0;y<28;y++){
		for(int x=0;x<28;x++){
			inframe(x,y) = inframe_001[y][x];
		}
	}

	tr.set_filter(filter_5x5_001);
	tr.in = inframe;
	tr.Proc();
	tr.view_filter();
	tr.view_conv();
	tr.view_out();

	return 0;
}

//-------------------------------------------------------------------------------------
// ディープラーニング試験(1)
//-------------------------------------------------------------------------------------
int test_03(void){
	typedef ap_fixed<8, 2, AP_RND, AP_SAT> T;
	const int SIZE1 = 20;
	const int SIZE2 = 20;
	const int SIZE3 = 500;
	const int SIZE4 = 10;
	const int WINDOW_SIZE = 5*5;

	T value;
	T L1_filter[SIZE1][WINDOW_SIZE];
	T L1_bias  [SIZE1];
	T L2_filter[SIZE2][SIZE1][WINDOW_SIZE];
	T L2_bias  [SIZE2][SIZE1];
	T L3_weight[SIZE3][SIZE2][7*7];
	T L3_bias  [SIZE3];
	T L4_weight[SIZE4][SIZE3];
	T L4_bias  [SIZE4];
	Matrix<28,28,T> inframe;
	Matrix<1,SIZE4,T> result;

	// 固定小数点確認
	for(int i=0;i<value.width;i++){
		value = 0;
		value[i] = 1;
		printf("ap_fixed<%d,%d> %16s = %f\n",value.width,value.iwidth, value.to_string(2).c_str(),(float)value);
	}
	value = 0;
	for(int i=0;i<value.width;i++){
		value[i] = 1;
		printf("ap_fixed<%d,%d> %16s = %f\n",value.width,value.iwidth, value.to_string(2).c_str(),(float)value);
	}

	// inframeロード
	for(int y=0;y<28;y++){
		for(int x=0;x<28;x++){
			inframe(x,y) = inframe_001[y][x]/9.0;
		}
	}
	// パラメータの初期化
	// Layer 1
	for(int i=0;i<SIZE1;i++){
		for(int j=0;j<WINDOW_SIZE;j++){
			L1_filter[i][j] = (j==WINDOW_SIZE/2)? 1 : 0;
		}
		L1_bias[i] = 0;
	}
	// Layer 2
	for(int i=0;i<SIZE2;i++){
		for(int j=0;j<SIZE1;j++){
			for(int k=0;k<WINDOW_SIZE;k++){
				L2_filter[i][j][k] = (k==WINDOW_SIZE/2)? 1 : 0;
			}
			L2_bias[i][j] = 0;
		}
	}
	// Layer 3
	for(int i=0;i<SIZE3;i++){
		for(int j=0;j<SIZE2;j++){
			for(int k=0;k<7*7;k++){
				L3_weight[i][j][k] = 1;
			}
		}
		L3_bias[i] = 0;
	}
	// Layer 4
	for(int i=0;i<SIZE4;i++){
		for(int j=0;j<SIZE3;j++){
			L4_weight[i][j] = 1;
		}
		L4_bias[i] = 0;
	}

	result = convolution_nn<28,28 ,5,5 ,SIZE1,SIZE2,SIZE3,SIZE4 ,T>
	         (inframe, L1_filter, L1_bias, L2_filter, L2_bias ,L3_weight ,L3_bias ,L4_weight ,L4_bias);

	return 0;
}

//-------------------------------------------------------------------------------------
// MNISTファイル読み込み試験
//-------------------------------------------------------------------------------------
int test_04(void){
	u8_t inframe[28*28];
	vector< vector<unsigned char> > image;
	vector<unsigned char> label;
	image = mnist.images();
	label = mnist.labels();

	for(int i=0; i<label.size() && i<5; i++){
		printf("%3d: %d\n",i ,(int)label.at(i));
		for(int y=0;y<mnist.height();y++){
			for(int x=0;x<mnist.width();x++){
				unsigned char pix = image[i][y*mnist.width()+x];
				if(pix==0x00){
					printf(" ..");
				} else {
					printf(" %02x",pix);
				}
				inframe[y*28+x] = pix;
			}
			printf("\n");
		}
		printf("\n");
	}

	raw_internal_t L1_filter[SIZE1][WINDOW_SIZE];        // 20x(5x5)=500
	raw_internal_t L1_bias  [SIZE1];                     // =20
	raw_internal_t L2_filter[SIZE2][SIZE1][WINDOW_SIZE]; // 20x20x(5x5)=10,000
	raw_internal_t L2_bias  [SIZE2][SIZE1];              // 20x20=400
	raw_internal_t L3_weight[SIZE3][SIZE2][7*7];         // 500x20x(7x7)=490,000
	raw_internal_t L3_bias  [SIZE3];                     // =500
	raw_internal_t L4_weight[SIZE4][SIZE3];              // 10x500=1,000
	raw_internal_t L4_bias  [SIZE4];                     // =10
	func_01_result_t result;

	result = func_01(inframe,L1_filter,L1_bias,L2_filter,L2_bias,L3_weight,L3_bias,L4_weight,L4_bias);
	return 0;
}

//-------------------------------------------------------------------------------------
// ディープラーニング試験(2)
//-------------------------------------------------------------------------------------
int test_05(void){
	typedef ap_fixed<9, 2, AP_RND, AP_SAT> T;
	const int SIZE1 = 20;
	const int SIZE2 = 20;
	const int SIZE3 = 500;
	const int SIZE4 = 10;
	const int WINDOW_SIZE = 5*5;

	T L1_filter[SIZE1][WINDOW_SIZE];
	T L1_bias  [SIZE1];
	T L2_filter[SIZE2][SIZE1][WINDOW_SIZE];
	T L2_bias  [SIZE2][SIZE1];
	T L3_weight[SIZE3][SIZE2][7*7];
	T L3_bias  [SIZE3];
	T L4_weight[SIZE4][SIZE3];
	T L4_bias  [SIZE4];
	Matrix<28,28,T> inframe;

	// inframeロード
	for(int y=0;y<28;y++){
		for(int x=0;x<28;x++){
			inframe(x,y) = inframe_001[y][x]/9.0;
		}
	}
	// パラメータの初期化
	// Layer 1
	for(int i=0;i<SIZE1;i++){
		for(int j=0;j<WINDOW_SIZE;j++){
			L1_filter[i][j] = RAND_FLOAT(-1.0, 1.0);
		}
		L1_bias[i] = 0;
	}
	// Layer 2
	for(int i=0;i<SIZE2;i++){
		for(int j=0;j<SIZE1;j++){
			for(int k=0;k<WINDOW_SIZE;k++){
				L2_filter[i][j][k] = RAND_FLOAT(-1.0, 1.0);
			}
			L2_bias[i][j] = 0;
		}
	}
	// Layer 3
	for(int i=0;i<SIZE3;i++){
		for(int j=0;j<SIZE2;j++){
			for(int k=0;k<7*7;k++){
				L3_weight[i][j][k] = RAND_FLOAT(-1.0, 1.0);
			}
		}
		L3_bias[i] = 0;
	}
	// Layer 4
	for(int i=0;i<SIZE4;i++){
		for(int j=0;j<SIZE3;j++){
			L4_weight[i][j] = RAND_FLOAT(-1.0, 1.0);
		}
		L4_bias[i] = 0;
	}

	Matrix<1,SIZE4,T> result;
	/*
	result = convolution_nn<28,28 ,5,5 ,SIZE1,SIZE2,SIZE3,SIZE4 ,T>
	         (inframe, L1_filter, L1_bias, L2_filter, L2_bias ,L3_weight ,L3_bias ,L4_weight ,L4_bias);
	*/
	result.view_float("Result");
	return 0;
}






