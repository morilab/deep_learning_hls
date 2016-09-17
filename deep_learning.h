#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Matrix.h"

// Feed-forward Neural Network
template <const int X, const int Y, typename T>
class relu_perceptron_fnn {
public:
	// In/Out Data
	Matrix<1,X,T> in;
	Matrix<1,Y,T> out;

	// Parameter
	Matrix<X,Y,T> weight;
	Matrix<1,Y,T> bias;

	// Calculate
	virtual Matrix<1,Y,T> run(Matrix<1,X,T> x){
		u = weight*x+bias;
		for(int y=0;y<Y;y++){
			z.v[0][y] = actfunc(u(0,y));
		}
		return z;
	}
	void run(){
		out = run(in);
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

	relu_perceptron_fnn(){
		// constructor
	}
	virtual ~relu_perceptron_fnn(){
		// destructor
	}

protected:
	Matrix<1,Y,T> u; //
	Matrix<1,Y,T> z; // Output
	virtual T actfunc(const T& in){ // Activation function(活性化関数)
		// rectifier(正規化線形関数)
		if(in<0){
			return 0;
		} else {
			return in;
		}
	}
};

template <const int X, const int Y, typename T>
class softmax_perceptron_fnn : public relu_perceptron_fnn<X,Y,T> {
public:
	softmax_perceptron_fnn(){
		// constructor
		sum_exp_u = 1.0;
	}
	virtual ~softmax_perceptron_fnn(){
		// destructor
	}

	virtual Matrix<1,Y,T> run(Matrix<1,X,T> x){
		this->u = this->weight*x+this->bias;
		sum_exp_u = 0.0;
		for(int y=0;y<Y;y++){
			sum_exp_u += exp((float)(this->u(0,y)));
		}
		for(int y=0;y<Y;y++){
			this->z(0,y) = actfunc(this->u(0,y));
		}
		return this->z;
	}

	void run(){
		this->out = run(this->in);
	}
private:
	float sum_exp_u;

protected:
	virtual T actfunc(const T& in){ // Activation function(活性化関数)
		// SoftMax(ソフトマックス)
		return exp((float)in)/sum_exp_u;
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
				filter(x,y)=1;
		    }
		}
		for(int y=0;y<IN_Y;y++){
			for(int x=0;x<IN_X;x++){
				conv(x,y)=0;
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
		filter.view_float("Filter");
	}

	void view_conv(){
		conv.view_float("Conv");
	}

	void view_out(){
		out.view_float("Out");
	}

protected:
	Matrix<CNV_X ,CNV_Y ,T> filter;
	Matrix<IN_X  ,IN_Y  ,T> conv;
	T bias;
	virtual T actfunc(const T& in){ // Activation function(活性化関数)
		// rectifier(正規化線形関数)
		if(in<0){
			return 0;
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
	relu_perceptron_fnn<X,Y,T> fnn;
	Matrix<1,X,T> mat_x;

	for(int i=0;i<Y;i++){
		fnn.bias(0,i)=bias[i];
		for(int j=0;j<X;j++){
			mat_x(0,j)=x[j];
			fnn.weight(j,i)=weight[i][j];
		}
	}
	return fnn.run(mat_x);
}

//-------------------------------------------------------------------
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
