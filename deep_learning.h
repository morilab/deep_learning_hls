#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Matrix.h"

template <const int X, const int Y, typename T>
class c_fnn {
public:
	Matrix<X,Y,T> weight;
	Matrix<1,Y,T> bias;
	Matrix<1,Y,T> run(Matrix<1,X,T> x) {
		u = weight*x+bias;
		return u;
	}

private:
	Matrix<1,Y,T> u;
};


template <const int X, const int Y, typename T>
Matrix<1,Y,T> deep_learning(Matrix<1,X,T> x, Matrix<X,Y,T> weight, Matrix<1,Y,T> bias) {
	c_fnn<X,Y,T> fnn;
	fnn.weight = weight;
	fnn.bias   = bias;
	return fnn.run(x);
}
