#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class MNIST{
public:
	MNIST(){
		// constructor
	}
	~MNIST(){
		// destructor
	}

	bool read_images(std::string filename);
	bool read_labels(std::string filename);
	std::vector<std::vector<unsigned char> > images();
	std::vector<unsigned char> labels();
	int width() {return row;};
	int height(){return col;};

private:
	int desc, num, row, col;
	std::vector< std::vector<unsigned char> > image;
	std::vector< unsigned char > label;
	int rev_int(unsigned char c[4]);
};
