//---------------------------------------------------------------
// SoftMax Perceptron (èoóÕëwóp)
//---------------------------------------------------------------
template <const int X, const int Y, typename T>
class softmax_perceptron_fnn : public perceptron_fnn<X,Y,T> {
	typedef perceptron_fnn<X,Y,T> base;
public:
	Matrix<X,Y,T> weight;
	Matrix<1,Y,T> bias;
	softmax_perceptron_fnn();
	virtual ~softmax_perceptron_fnn(){}
	void run();
protected:
	T actfunc(const T& in); // => SoftMax
private:
	Matrix<1,Y,T> u;
	float sum_exp_u;
};

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
softmax_perceptron_fnn<X,Y,T>::softmax_perceptron_fnn() {
	sum_exp_u = 1.0;
}

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
void softmax_perceptron_fnn<X,Y,T>::run() {
	u = weight*base::in+bias;
	sum_exp_u = 0.0;
	for(int y=0;y<Y;y++){
		sum_exp_u += exp((float)(u(0,y)));
	}
	for(int y=0;y<Y;y++){
		base::out(0,y) = actfunc(u(0,y));
	}
}

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
T softmax_perceptron_fnn<X,Y,T>::actfunc(const T& in) {
	return exp((float)in)/sum_exp_u;
}
