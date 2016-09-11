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
	virtual int errfunc(Matrix<1,Y,T> d){ // error function(Œë·ŠÖ”)
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
	virtual T actfunc(const T& in){ // Activation function(Šˆ«‰»ŠÖ”)
		// rectifier(³‹K‰»üŒ`ŠÖ”)
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
