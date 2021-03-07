#include <iostream>
#include <fstream>
#include <cassert>
#include <random>
#include <string>


int main(int argc, char* argv[]) {

	assert(argc == 3); int l = std::stoi(argv[1]), max_nm = std::stoi(argv[2]);
	using T = unsigned long;

	std::vector<std::pair<T,T>> Jacobian(l);

	std::random_device r;
	std::default_random_engine g(r());
	std::uniform_int_distribution<T> dnm(1, max_nm);

	T m = dnm(g), n = dnm(g);
	
	//Filling up the dimensions for the Jacobians
	Jacobian[0] = std::make_pair(m, n);
	for (int i = 1; i < l; i++) {
		m = n;
		n = dnm(g);

        Jacobian[i] = std::make_pair(m, n);
	}

	std::cout << l << std::endl;
	for (auto itr = Jacobian.crbegin(); itr != Jacobian.crend(); ++itr) {
		std::cout << itr->first << " " << itr->second << std::endl;
	}

	return 0;
}
