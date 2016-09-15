#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Matrix.h"

// Feed-forward Neural Network
template <const int X, const int Y, typename T>
class c_fnn {
public:
	// Parameter
	Matrix<X,Y,T> weight;
	Matrix<1,Y,T> bias;

	// Calculate
	Matrix<1,Y,T> run(Matrix<1,X,T> x){
		u = weight*x+bias;
		for(int y=0;y<Y;y++){
			z.v[0][y] = actfunc(u(0,y));
		}
		return z;
	}
	virtual int errfunc(Matrix<1,Y,T> d){ // error function(誤差関数)
		T err;
		err = (T)0;
		for(int j=0;j<Y;j++){
			T dt = d(0,j)-z(0,j);
			err += dt*dt;
		}
		return err/2;
	}

	c_fnn(){
		// constructor
	}
	virtual ~c_fnn(){
		// destructor
	}

private:
	Matrix<1,Y,T> u; //
	Matrix<1,Y,T> z; // Output
	virtual T actfunc(const T& in){ // Activation function(活性化関数)
		// rectifier(正規化線形関数)
		if(in<0){
			return (T)0;
		} else {
			return in;
		}
	}
};

//-------------------------------------------------------------------
// convolution perceptron class
//-------------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
class convolution_perceptron {
public:
	Matrix<IN_X  ,IN_Y  ,T> in;
	Matrix<IN_X/2,IN_Y/2,T> out;

	convolution_perceptron(){
		for(int y=0;y<CNV_Y;y++){
			for(int x=0;x<CNV_X;x++){
				filter(x,y)=(T)1;
		    }
		}
		for(int y=0;y<IN_Y;y++){
			for(int x=0;x<IN_X;x++){
				conv(x,y)=(T)0;
			}
		}
		bias = (T)0;
	}

	virtual ~convolution_perceptron(){
	}

	void Convolution(){
		// 畳み込み処理
		for(int y=0;y<IN_Y;y++){
			for(int x=0;x<IN_X;x++){
				// 畳み込み演算
				T pix = 0;
				for(int dy=-(CNV_Y-1)/2;dy<(CNV_Y+1)/2;dy++){
					for(int dx=-(CNV_X-1)/2;dx<(CNV_X+1)/2;dx++){
						if(x+dx>=0 && y+dy>=0 && x+dx<IN_X && y+dy<IN_Y){
							pix += in(x+dx,y+dy) * filter(dx+(CNV_X-1)/2,dy+(CNV_Y-1)/2);
						}
					}
				}
				conv(x,y) += pix;
			}
		}
	}

	void Activation(){
		for(int y=0;y<IN_Y;y++){
			for(int x=0;x<IN_X;x++){
				conv(x,y) += bias; // バイアス処理
				conv(x,y) = actfunc(conv(x,y)); // 活性化関数
			}
		}
	}

	void MaxPooling(){
		// 2x2 MAXプーリング
		for(int y=0;y<IN_Y/2;y++){
			for(int x=0;x<IN_X/2;x++){
				T max_x0 = (conv(x*2+0,y*2)>conv(x*2+0,y*2+1)) ? conv(x*2+0,y*2) : conv(x*2+0,y*2+1);
				T max_x1 = (conv(x*2+1,y*2)>conv(x*2+1,y*2+1)) ? conv(x*2+1,y*2) : conv(x*2+1,y*2+1);
				out(x,y) = (max_x0>max_x1) ? max_x0 : max_x1;
			}
		}
	}

	void clear_conv(){
		for(int y=0;y<IN_Y;y++){
			for(int x=0;x<IN_X;x++){
				conv(x,y) = 0;
			}
		}
	}

	void Proc(){
		clear_conv();
		Convolution();	// 畳み込み処理
		Activation();   // バイアス＋活性化処理
		MaxPooling();	// MAXプーリング
	}

	void set_filter(const T val[CNV_X*CNV_Y]){
		for(int y=0;y<CNV_Y;y++){
			for(int x=0;x<CNV_X;x++){
				filter(x,y)=val[y*CNV_X+x];
			}
		}
	}

	void set_bias(T val){
		bias = val;
	}

	void view_filter(){
		printf("filter = {\n");
		for(int y=0;y<CNV_Y;y++){
			printf("  {");
			for(int x=0;x<CNV_X;x++){
				printf(" %3d",filter(x,y));
			}
			printf(" }\n");
		}
		printf("};\n\n");
	}

	void view_conv(){
		printf("conv = {\n");
		for(int y=0;y<IN_Y;y++){
			printf("  {");
			for(int x=0;x<IN_X;x++){
				printf(" %3d",conv(x,y));
			}
			printf(" }\n");
		}
		printf("};\n\n");
	}

	void view_out(){
		printf("out = {\n");
		for(int y=0;y<IN_Y/2;y++){
			printf("  {");
			for(int x=0;x<IN_X/2;x++){
				printf(" %3d",out(x,y));
			}
			printf(" }\n");
		}
		printf("};\n\n");
	}

private:
	Matrix<CNV_X ,CNV_Y ,T> filter;
	Matrix<IN_X  ,IN_Y  ,T> conv;
	T bias;
	virtual T actfunc(const T& in){ // Activation function(活性化関数)
		// rectifier(正規化線形関数)
		if(in<0){
			return (T)0;
		} else {
			return in;
		}
	}

};

//-------------------------------------------------------------------
// wrapper for VivadoHLS
//-------------------------------------------------------------------
template <const int X, const int Y, typename T>
Matrix<1,Y,T> deep_learning(T x[X], T weight[Y][X], T bias[Y]) {
	c_fnn<X,Y,T> fnn;
	Matrix<1,X,T> mat_x;
	Matrix<X,Y,T> mat_weight;

	for(int i=0;i<Y;i++){
		fnn.bias.v[0][i]=bias[i];
		for(int j=0;j<X;j++){
			mat_x.v[0][j]=x[j];
			fnn.weight.v[j][i]=weight[i][j];
		}
	}
	return fnn.run(mat_x);
}

template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, const int SIZE1, const int SIZE2, typename T>
Matrix<1,IN_X*IN_Y*SIZE2/16,T> convolution_nn(Matrix<IN_X,IN_Y,T> inframe, T filter_L1[SIZE1][CNV_X*CNV_Y], T bias_L1[SIZE1], T filter_L2[SIZE2][SIZE1][CNV_X*CNV_Y], T bias_L2[SIZE2][SIZE1]) {
	convolution_perceptron<IN_X  ,IN_Y  ,CNV_X,CNV_Y,T> perceptron_L1[SIZE1];
	convolution_perceptron<IN_X/2,IN_Y/2,CNV_X,CNV_Y,T> perceptron_L2[SIZE2];
	Matrix<1,IN_X*IN_Y*SIZE2/16,T> connect;

	for(int i=0;i<SIZE1;i++){
		perceptron_L1[i].in = inframe;
		perceptron_L1[i].set_bias(bias_L1[i]);
		perceptron_L1[i].set_filter(filter_L1[i]);
		perceptron_L1[i].Proc();
	}
	for(int j=0;j<SIZE2;j++){
		perceptron_L2[j].clear_conv();
		for(int i=0;i<SIZE1;i++){
			perceptron_L2[j].in = perceptron_L1[i].out;
			perceptron_L2[j].set_bias(bias_L2[j][i]);
			perceptron_L2[j].set_filter(filter_L2[j][i]);
			perceptron_L2[j].Conversion();
		}
		perceptron_L2[j].Activation();
		perceptron_L2[j].MaxPooling();
		for(int y=0;y<IN_Y/4;y++){
			for(int x=0;x<IN_X/4;x++){
				connect(1,(IN_X*IN_Y/16)*j+y*(IN_X/4)+x) = perceptron_L2[j].out(x,y);
			}
		}
	}
	return connect;
}
