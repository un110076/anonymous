
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <chrono>
#include "Eigen/Eigen/Dense"



template<typename T>
struct Triplet {

	T x; T y; T z;

	// constructor
	Triplet()
		: x(0), y(0), z(0)
	{}

	// destructor
	~Triplet() {};
};



template<typename T>
struct Cost_Triplet {

	T cost;
	T split_pos;
	Triplet<T> dim;

	// constructor
	Cost_Triplet()
		: cost(0), split_pos(0)
	{}

	// destructor
	~Cost_Triplet() {};

};



template<typename T>
class Tensor {
public:

	std::vector<std::vector<std::vector<float>>> data;
	Tensor() {};

	// Randomize the elements of Tensor
	void Random(int x, int y, int z) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(1.0, 4.0);

		data.resize(x);
		for (int i = 0; i < x; i++) {
			data[i].resize(y);
			for (int j = 0; j < y; j++) {
				data[i][j].resize(z);
				for (int k = 0; k < z; k++) {
					data[i][j][k] = dis(gen);
				}
			}
		}
	}

	// Initialize the Tensors
	void init(int x, int y, int z) {
		data.resize(x);
		for (int i = 0; i < x; i++) {
			data[i].resize(y);
			for (int j = 0; j < y; j++) {
				data[i][j].resize(z);
			}
		}
	}

	// Assignment operator for Tensors 
	void operator=(const Tensor<T>& othertensor) {
		this->init(othertensor.data.size(), othertensor.data[0].size(), othertensor.data[0][0].size());
		for (int i = 0; i < othertensor.data.size(); i++) {
			for (int j = 0; j < othertensor.data[i].size(); j++) {
				for (int k = 0; k < othertensor.data[i][j].size(); k++) {
					this->data[i][j][k] = othertensor.data[i][j][k];
				}
			}
		}
	}

	// Add two Tensors
	Tensor<T> operator+(const Tensor<T>& othertensor) {
		Tensor<T> result;
		result.data.resize(this->data.size());
		for (int i = 0; i < othertensor.data.size(); i++) {
			result.data[i].resize(this->data[i].size());
			for (int j = 0; j < othertensor.data[i].size(); j++) {
				result.data[i][j].resize(this->data[i][j].size());
				for (int k = 0; k < othertensor.data[i][j].size(); k++) {
					result.data[i][j][k] = this->data[i][j][k] + othertensor.data[i][j][k];
				}
			}
		}
		return result;
	}

	~Tensor() {};
};



template<typename T>
T Jacobian_cost(const std::vector<Triplet<T>>& Jac,
	std::vector<std::vector<Cost_Triplet<T>>>& CJ) {

	T p = Jac.size();

	for (int j = 0; j < p; j++) {
		for (int i = j; i >= 0; i--) {
			if (i == j) {
				CJ[j][i].cost = 0;
				CJ[j][i].split_pos = 0;
				CJ[j][i].dim = Jac[j];
			}
			else {
				for (int k = i + 1; k <= j; k++) {
					T fma = Jac[j].x * Jac[k - 1].x * Jac[i].y;
					T cost_Jac = CJ[j][k].cost + CJ[k - 1][i].cost + fma;
					if (k == i + 1 || cost_Jac < CJ[j][i].cost) {
						CJ[j][i].cost = cost_Jac;
						CJ[j][i].split_pos = k;
					}
				}
				// calculating the dimension of the resulting Jacobian subchain
				CJ[j][i].dim.x = Jac[j].x;
				CJ[j][i].dim.y = Jac[i].y;
				CJ[j][i].dim.z = 1;
			}
		}
	}

	return CJ[p - 1][0].cost;

}



template<typename T>
void Jacobian_Product_evaluation(const std::vector<std::vector<Cost_Triplet<T>>>& CJ,
	const std::vector<Eigen::MatrixXf>& Jacobian_factors,
	std::vector<std::vector<Eigen::MatrixXf>>& Jacobian_products) {


	T p = Jacobian_factors.size();

	for (int j = 0; j < p; j++) {
		for (int i = j; i >= 0; i--) {
			if (i == j) {
				Jacobian_products[j][i] = Jacobian_factors[j];   
			}
			else {
				Jacobian_products[j][i] = Jacobian_products[j][CJ[j][i].split_pos] *
					Jacobian_products[CJ[j][i].split_pos - 1][i];      
			}
		}
	}
}


// Matrix-Tensor Product
template<typename T>
Tensor<T> MatXTens(const Eigen::MatrixXf& A, const Tensor<T>& B) {

	Tensor<T> result;
	result.init(A.rows(), B.data[0].size(), B.data[0][0].size());

	for (int k = 0; k < B.data[0][0].size(); k++) {
		for (int i = 0; i < A.rows(); i++) {
			for (int j = 0; j < B.data[0].size(); j++) {
				for (int s = 0; s < A.cols(); s++) {
					result.data[i][j][k] += A(i, s) * B.data[s][j][k];
				}
			}
		}
	}
	return result;
}


// Tensor-Matrix Product
template<typename T>
Tensor<T> TensXMat(const Tensor<T>& B, const Eigen::MatrixXf& A) {

	Tensor<T> result;
	result.init(B.data.size(), B.data[0].size(), A.cols());

	for (int j = 0; j < B.data[0].size(); j++) {
		for (int i = 0; i < B.data.size(); i++) {
			for (int k = 0; k < A.cols(); k++) {
				for (int s = 0; s < B.data[0][0].size(); s++) {
					result.data[i][j][k] += B.data[i][j][s] * A(s, k);
				}
			}
		}
	}
	return result;
}


// Tensor-Matrix dyadic product
template<typename T>
Tensor<T> TensXMatCross(const Tensor<T>& B, const Eigen::MatrixXf& A) {

	Tensor<T> result;
	result.init(B.data.size(), B.data[0][0].size(), A.cols());

	for (int k = 0; k < B.data[0][0].size(); k++) {
		for (int i = 0; i < B.data.size(); i++) {
			for (int j = 0; j < A.cols(); j++) {
				for (int s = 0; s < B.data[0].size(); s++) {
					result.data[i][k][j] += B.data[i][s][k] * A(s, j);
				}
			}
		}
	}
	return result;
}



template<typename T>
void Hessian_product_evaluation(const std::vector<std::vector<Eigen::MatrixXf>>& Jacobian_products,
	const std::vector<Tensor<T>>& Hessian_factors,
	std::vector<std::vector<Tensor<T>>>& Hessian_products,
	const std::vector<Triplet<T>>& B) {

	for (int i = 0; i < Hessian_factors.size(); i++) { Hessian_products[i][i] = Hessian_factors[i]; }
	for (auto& itr : B) {
		Hessian_products[itr.x][itr.z] = MatXTens(Jacobian_products[itr.x][itr.y],
			Hessian_products[itr.y - 1][itr.z]) +
			TensXMatCross(TensXMat(Hessian_products[itr.x][itr.y], Jacobian_products[itr.y - 1][itr.z]),
				Jacobian_products[itr.y - 1][itr.z]);
	}

}



template<typename T>
void Bracket_right(const std::vector<Eigen::MatrixXf>& Jacobian_factors,
	const std::vector<Tensor<T>>& Hessian_factors,
	std::vector<std::vector<Tensor<T>>>& Hess_prod) {

	T p = Jacobian_factors.size();
	Tensor<T> Hess_prod_1;
	Tensor<T> Hess_prod_2;
	Eigen::MatrixXf Jac_prod;


	// Bracketing from right (A(B(C(D.....(X)))))
	Hess_prod[0][0] = Hessian_factors[0];
	for (int j = 1; j < p; j++) {

		// First Resultant Hessian term
		Hess_prod_1 = MatXTens(Jacobian_factors[j], Hess_prod[j - 1][0]);

		// Second Resultant Hessian term
		Jac_prod = Jacobian_factors[0];
		for (int r = 1; r < j; r++) {Jac_prod = Jacobian_factors[r] * Jac_prod;}
		Hess_prod_2 = TensXMat(Hessian_factors[j], Jac_prod);
		Hess_prod_2 = TensXMatCross(Hess_prod_2, Jac_prod);


		// Store for use in following iterations
		Hess_prod[j][0] = Hess_prod_1 + Hess_prod_2;
	}

}



template<typename T>
void Bracket_left(const std::vector<Eigen::MatrixXf>& Jacobian_factors,
	const std::vector<Tensor<T>>& Hessian_factors,
	std::vector<std::vector<Tensor<T>>>& Hess_prod) {

	T p = Jacobian_factors.size();
	Tensor<T> Hess_prod_1;
	Tensor<T> Hess_prod_2;
	Eigen::MatrixXf Jac_prod;

	// Bracketing from left ((((A)B)C)D.....X)
	Hess_prod[p - 1][p - 1] = Hessian_factors[p - 1];
	for (int j = p - 1; j > 0; j--) {

		// First Resultant Hessian term
		Jac_prod = Jacobian_factors[p - 1];
		for (int r = p - 1; r > j; r--) { Jac_prod = Jac_prod * Jacobian_factors[r - 1]; }
		Hess_prod_1 = MatXTens(Jac_prod, Hessian_factors[j - 1]);

		// Second Resultant Hessian term
		Hess_prod_2 = TensXMat(Hess_prod[p - 1][j], Jacobian_factors[j - 1]);
		Hess_prod_2 = TensXMatCross(Hess_prod_2, Jacobian_factors[j - 1]);


		// Store for use in following iterations
		Hess_prod[p - 1][j - 1] = Hess_prod_1 + Hess_prod_2;
	}

}



int main(int argc, char* argv[]) {

	assert(argc == 3); std::ifstream in_1(argv[1]); std::ifstream in_2(argv[2]);
	using T = unsigned long;


	T p; in_1 >> p; assert(p > 0);            // This gives us the order of compositeness of the hessian 

	std::vector<Triplet<T>> Jac(p);           // Jacobian vector
	std::vector<Triplet<T>> Hess(p);          // Hessian vector
	std::vector<Triplet<T>> B(p - 1);

	// Costs for Jacobian vector
	std::vector<std::vector<Cost_Triplet<T>>> CJ(p, std::vector<Cost_Triplet<T>>(p));

	// Input vector for Jacobian and Hessian size
	for (unsigned int i = 0; i < p; i++) {
		in_1 >> Jac[i].x >> Jac[i].y;
		Hess[i].x = Jac[i].x;
        Hess[i].y = Jac[i].y;
        Hess[i].z = Jac[i].y;
		Jac[i].z = 1;
	}

	// Input vector for optimal Bracketing order of the Hessian
	for (unsigned int i = 0; i < p - 1; i++) {
		in_2 >> B[i].x >> B[i].y >> B[i].z;
	}

	std::vector<Eigen::MatrixXf> Jacobian_factors;
	std::vector<Tensor<T>> Hessian_factors(p);
	std::vector<std::vector<Eigen::MatrixXf>> Jacobian_products(p, std::vector<Eigen::MatrixXf>(p));
	std::vector<std::vector<Tensor<T>>> Hessian_products(p, std::vector<Tensor<T>>(p));
	for (unsigned int i = 0; i < p; i++) {
		Jacobian_factors.push_back(Eigen::MatrixXf::Random(Jac[i].x, Jac[i].y));
		Hessian_factors[i].Random(Hess[i].x, Hess[i].y, Hess[i].z);
	}

	std::cout << "Elapsed time (in microseconds):" << std::endl;
	// Brute force way of computing the FMA for the Hessian chain 
	// (Bracket from right and left)
	std::vector<std::vector<Tensor<T>>> Hess_prod(p, std::vector<Tensor<T>>(p));
	auto t1 = std::chrono::high_resolution_clock::now();
	Bracket_left(Jacobian_factors, Hessian_factors, Hess_prod);
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	std::cout << "left bracketing: " << duration_1 << std::endl;

	auto t3 = std::chrono::high_resolution_clock::now();
	Bracket_right(Jacobian_factors, Hessian_factors, Hess_prod);
	auto t4 = std::chrono::high_resolution_clock::now();
	auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
	std::cout << "right bracketing: " << duration_2 << std::endl;




	// Dynamic Programming Algorithm
	Jacobian_cost(Jac, CJ);
	auto t5 = std::chrono::high_resolution_clock::now();
	Jacobian_Product_evaluation(CJ, Jacobian_factors, Jacobian_products);
	Hessian_product_evaluation(Jacobian_products, Hessian_factors, Hessian_products, B);
	auto t6 = std::chrono::high_resolution_clock::now();
	auto duration_3 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
	std::cout << "optimized bracketing: " << duration_3 << std::endl;


	return 0;
}
