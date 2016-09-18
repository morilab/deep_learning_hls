//---------------------------------------------------------------
// ReLU Perceptron
//---------------------------------------------------------------
template <const int X, const int Y, typename T>
class relu_perceptron_fnn : public perceptron_fnn<X,Y,T> {
	typedef perceptron_fnn<X,Y,T> base;
public:
	Matrix<X,Y,T> weight;   // ウエイト
	Matrix<1,Y,T> bias;     // バイアス
	relu_perceptron_fnn(){}
	virtual ~relu_perceptron_fnn(){}
	void run();
protected:
	T actfunc(const T& in); // => ReLU
private:
	Matrix<1,Y,T> u;
};

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
void relu_perceptron_fnn<X,Y,T>::run() {
	u = weight*base::in+bias;
	for(int y=0;y<Y;y++){
		base::out(0,y) = actfunc(u(0,y));
	}
}

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
T relu_perceptron_fnn<X,Y,T>::actfunc(const T& in) {
	if(in<0){
		return 0;
	} else {
		return in;
	}
}
