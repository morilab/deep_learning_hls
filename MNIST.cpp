#include "MNIST.h"

int MNIST::rev_int (unsigned char c[4])
{
	int res = 0;
    res += (int)c[0]<<(8*3);
    res += (int)c[1]<<(8*2);
    res += (int)c[2]<<(8*1);
    res += (int)c[3]<<(8*0);
    return res;
}

bool MNIST::read_images(std::string filename){
	std::ifstream fp(filename.c_str(), std::ios::in | std::ios::binary);
	if(!fp) return false;

	unsigned char int_read_c[4];
	int magic_number;

	fp.read(reinterpret_cast<char*>(int_read_c), sizeof(int_read_c));
	magic_number = rev_int(int_read_c);
	fp.read(reinterpret_cast<char*>(int_read_c), sizeof(int_read_c));
	num = rev_int(int_read_c);
	fp.read(reinterpret_cast<char*>(int_read_c), sizeof(int_read_c));
	row = rev_int(int_read_c);
	fp.read(reinterpret_cast<char*>(int_read_c), sizeof(int_read_c));
	col = rev_int(int_read_c);

	std::vector<unsigned char> read_vector(row*col);
	while(true){
		fp.read(reinterpret_cast<char*>(&(read_vector[0])), sizeof(unsigned char)*row*col);
		if(fp.eof()) break;
		image.push_back(read_vector);
	}
	return true;
}

bool MNIST::read_labels(std::string filename){
	std::ifstream fp(filename.c_str(), std::ios::in | std::ios::binary);
	if(!fp) return false;

	unsigned char int_read_c[4];
	int magic_number;

	fp.read(reinterpret_cast<char*>(int_read_c), sizeof(int_read_c));
	magic_number = rev_int(int_read_c);
	fp.read(reinterpret_cast<char*>(int_read_c), sizeof(int_read_c));
	num = rev_int(int_read_c);

	char read_c;
	while(true){
		fp.read(&read_c, sizeof(read_c));
		if(fp.eof()) break;
		label.push_back(read_c);
	}
	return true;
}

std::vector<std::vector<unsigned char> > MNIST::images(){
	return image;
}
std::vector<unsigned char> MNIST::labels(){
	return label;
}

