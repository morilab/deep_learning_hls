/* [Matrix]
 * 行列演算ライブラリ
 */

template <const int X, const int Y, typename T>
class Matrix {
public:
	T v[X][Y];

	// コンストラクタ
	Matrix() {
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				v[i][j] = (T)0;
			}
		}
	}


	// 演算子のオーバーライド
	Matrix<X,Y,T>& operator = (const Matrix<X,Y,T>& in){
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				this->v[i][j] = in.v[i][j];
			}
		}
		return *this;
	}
	Matrix<X,Y,T>  operator+(){
		return *this;
	}
	Matrix<X,Y,T>  operator-(){
		Matrix<X,Y,T> tmp;
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				tmp.v[i][j] =- v[i][j];
			}
		}
		return tmp;
	}
	Matrix<X,Y,T>& operator +=(const Matrix<X,Y,T>& in){
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				v[i][j] += in.v[i][j];
			}
		}
		return *this;
	}
	Matrix<X,Y,T>& operator -=(const Matrix<X,Y,T>& in){
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				v[i][j] -= in.v[i][j];
			}
		}
		return *this;
	}
	Matrix<X,Y,T>& operator *=(const T coef){
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				v[i][j] *= coef;
			}
		}
		return *this;
	}
	Matrix<X,Y,T>& operator /=(const T coef){
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				v[i][j] /= coef;
			}
		}
		return *this;
	}

	// 添え字演算子
	T* operator[] (const int i){
		return v[i];
	}
	T& operator() (const int i ,const int j){
		return v[i][j];
	}

private:

};

// 行列演算
template <const int X, const int Y, typename T>
inline Matrix<X,Y,T> operator+(const Matrix<X,Y,T>& in1 ,const Matrix<X,Y,T>& in2){
	Matrix<X,Y,T> tmp;
	for(int i=0;i<X;i++){
		for(int j=0;j<Y;j++){
			tmp.v[i][j] = in1.v[i][j] + in2.v[i][j];
		}
	}
	return tmp;
}

template <const int X, const int Y, typename T>
inline Matrix<1,Y,T> operator*(const Matrix<X,Y,T>& in1, const Matrix<1,X,T>& coef){
	Matrix<1,Y,T> tmp;
	for(int j=0;j<Y;j++){
		tmp(0,j)=0;
		for(int i=0;i<X;i++){
			tmp.v[0][j] += coef.v[0][i] * in1.v[i][j];
		}
	}
	return tmp;
}
