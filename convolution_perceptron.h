//-------------------------------------------------------------------
// convolution perceptron class
//-------------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
class convolution_perceptron {
public:
	Matrix<IN_X  ,IN_Y  ,T> in;
	Matrix<IN_X/2,IN_Y/2,T> out;
	Matrix<CNV_X ,CNV_Y ,T> filter;
	T bias;
	convolution_perceptron();
	virtual ~convolution_perceptron(){}
	void Convolution();
	void Activation();
	void MaxPooling();
	void clear_conv();
	void Proc();
	void set_filter(const T val[CNV_X*CNV_Y]);
	void set_bias(T val);
	void view_filter(){	filter.view_float("Filter"); }
	void view_conv()  {	conv.view_float("Conv");     }
	void view_out()   {	out.view_float("Out");       }
protected:
	T actfunc(const T& in);  // => ReLU
private:
	Matrix<IN_X,IN_Y,T> conv;
};

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::convolution_perceptron(){
	ConvolutionPerceptron_Init1_L1 :
	for(int y=0;y<CNV_Y;y++){
		ConvolutionPerceptron_Init1_L2 :
		for(int x=0;x<CNV_X;x++){
			filter(x,y) = 1;
	    }
	}
	ConvolutionPerceptron_Init2_L1 :
	for(int y=0;y<IN_Y;y++){
		ConvolutionPerceptron_Init2_L2 :
		for(int x=0;x<IN_X;x++){
			conv(x,y) = 0;
		}
	}
	bias = 0;
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::Convolution(){
	// 畳み込み処理
	ConvolutionPerceptron_Convolution_L1 :
	for(int y=0;y<IN_Y;y++){
		ConvolutionPerceptron_Convolution_L2 :
		for(int x=0;x<IN_X;x++){
			// 畳み込み演算
			T pix = 0;
			ConvolutionPerceptron_Convolution_L3 :
			for(int dy=-(CNV_Y-1)/2;dy<(CNV_Y+1)/2;dy++){
				ConvolutionPerceptron_Convolution_L4 :
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

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::Activation(){
	ConvolutionPerceptron_Activation_L1 :
	for(int y=0;y<IN_Y;y++){
		ConvolutionPerceptron_Activation_L2 :
		for(int x=0;x<IN_X;x++){
			conv(x,y) += bias; // バイアス処理
			conv(x,y) = actfunc(conv(x,y)); // 活性化関数
		}
	}
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::MaxPooling(){
	// 2x2 MAXプーリング
	ConvolutionPerceptron_MaxPooling_L1 :
	for(int y=0;y<IN_Y/2;y++){
		ConvolutionPerceptron_MaxPooling_L2 :
		for(int x=0;x<IN_X/2;x++){
			T max_x0 = (conv(x*2+0,y*2)>conv(x*2+0,y*2+1)) ? conv(x*2+0,y*2) : conv(x*2+0,y*2+1);
			T max_x1 = (conv(x*2+1,y*2)>conv(x*2+1,y*2+1)) ? conv(x*2+1,y*2) : conv(x*2+1,y*2+1);
			out(x,y) = (max_x0>max_x1) ? max_x0 : max_x1;
		}
	}
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::clear_conv(){
	ConvolutionPerceptron_ClearConv_L1 :
	for(int y=0;y<IN_Y;y++){
		ConvolutionPerceptron_ClearConv_L2 :
		for(int x=0;x<IN_X;x++){
			conv(x,y) = 0;
		}
	}
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::Proc(){
	clear_conv();
	Convolution();	// 畳み込み処理
	Activation();   // バイアス＋活性化処理
	MaxPooling();	// MAXプーリング
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::set_filter(const T val[CNV_X*CNV_Y]){
	ConvolutionPerceptron_SetFilter_L1 :
	for(int y=0;y<CNV_Y;y++){
		ConvolutionPerceptron_SetFilter_L2 :
		for(int x=0;x<CNV_X;x++){
			filter(x,y) = val[y*CNV_X+x];
		}
	}
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
void convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::set_bias(T val){
	bias = val;
}

//---------------------------------------------------------------
template <const int IN_X, const int IN_Y, const int CNV_X, const int CNV_Y, typename T>
T convolution_perceptron<IN_X,IN_Y,CNV_X,CNV_Y,T>::actfunc(const T& in){
	if(in<0){
		return 0;
	} else {
		return in;
	}
}
