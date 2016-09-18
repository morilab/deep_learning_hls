//---------------------------------------------------------------
// �P���p�[�Z�v�g�����@���N���X
//---------------------------------------------------------------
template <const int X, const int Y, typename T>
class perceptron_fnn {
public:
	Matrix<1,X,T> in;              // ���̓f�[�^
	Matrix<1,Y,T> out;             // �o�̓f�[�^
	perceptron_fnn(){}
	virtual ~perceptron_fnn(){}
	void run(){}                   // ���Z�̎��s
	int errfunc(Matrix<1,Y,T>& d); // error function(�덷�֐�)
protected:
	T actfunc(const T& in);        // Activation function(�������֐�)
};

//---------------------------------------------------------------
template <const int X, const int Y, typename T>
int perceptron_fnn<X,Y,T>::errfunc(Matrix<1,Y,T>& d){ // error function(�덷�֐�)
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
