//---------------------------------------------------------------
// 単純パーセプトロン　基底クラス
//---------------------------------------------------------------
template <const int X, const int Y, typename T>
class perceptron_fnn {
public:
	Matrix<1,X,T> in;              // 入力データ
	Matrix<1,Y,T> out;             // 出力データ
	perceptron_fnn(){}
	virtual ~perceptron_fnn(){}
	void run(){}                   // 演算の実行
	int errfunc(Matrix<1,Y,T>& d); // error function(誤差関数)
protected:
	T actfunc(const T& in);        // Activation function(活性化関数)
};

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
int perceptron_fnn<X,Y,T>::errfunc(Matrix<1,Y,T>& d){ // error function(誤差関数)
	int err;
	err = 0;
	for(int j=0;j<Y;j++){
		T dt = d(0,j)-out(0,j);
		err += dt*dt;
	}
	return err/2;
}

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
T perceptron_fnn<X,Y,T>::actfunc(const T& in){
	return in;
}
