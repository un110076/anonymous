
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include <numeric>


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
struct node {

	Triplet<T> info;
	node *left;
	node *right;

	// constructor
	node()
		:left(nullptr), right(nullptr)
	{}

	node(const int& info)
		:info(info), left(nullptr), right(nullptr)
	{}

	// member function
	int max_depth() const {
		const int left_depth = left ? left->max_depth() : 0;
		const int right_depth = right ? right->max_depth() : 0;
		return (left_depth > right_depth ? left_depth : right_depth) + 1;
	}

	// destructor
	~node() { delete left; delete right; }

};



template<typename T>
class BST {
	
	node<T>* root;

public:

	// constructor
	BST()
	{
		root = NULL;
	}

	// member functions
	int get_max_depth() const { return root ? root->max_depth() : 0; }
	node<T>* insert(node<T> *, node<T> *);
	void inorder(node<T> *, std::vector<Triplet<T>>&);
	void postorder(node<T> *);
	void givenlevel(node<T> *, int, std::ofstream&);
	void levelorder(node<T> *, std::ofstream&);
	void display(node<T> *, int);
	void clear() { delete root; root = nullptr; }

	// destructor
	~BST() { delete root; }

private:

	// data members
	std::vector<Triplet<T>> node_collector;

};



template<typename T>
node<T>* BST<T>::insert(node<T> *tree, node<T> *newnode) {
	if (root == NULL)
	{
		root = new node<T>;
		root->info = newnode->info;
		root->left = NULL;
		root->right = NULL;

		return root;
	}
	if (tree->info.y == newnode->info.y)
	{
		std::cout << "Element already in the tree" << std::endl;
		return tree;
	}
	if (tree->info.y > newnode->info.y)
	{
		if (tree->right != NULL) {
			insert(tree->right, newnode);
		}
		else {
			tree->right = newnode;
			(tree->right)->left = NULL;
			(tree->right)->right = NULL;

			return tree;
		}
	}
	else {
		if (tree->left != NULL) {
			insert(tree->left, newnode);
		}
		else {
			tree->left = newnode;
			(tree->left)->left = NULL;
			(tree->left)->right = NULL;

			return tree;
		}

	}
	
	return tree;
}


template<typename T>
void BST<T>::inorder(node<T> *ptr, std::vector<Triplet<T>>& node_collector) {
	if (root == NULL)
	{
		std::cout << "Tree is empty" << std::endl;
		return;
	}
	if (ptr != NULL)
	{
		inorder(ptr->left, node_collector);
		node_collector.push_back(ptr->info);
		inorder(ptr->right, node_collector);
	}
}


template<typename T>
void BST<T>::postorder(node<T> *ptr)
{
	if (ptr == NULL)
		return;

	// first recur on left subtree 
	postorder(ptr->left);

	// then recur on right subtree 
	postorder(ptr->right);

	// now deal with the node 
	std::cout << ptr->info.y << " ";
}


template<typename T>
void BST<T>::givenlevel(node<T> *ptr, int level, std::ofstream& solution) {

	if (ptr == NULL)
		return;
	if (level == 1)
		solution << ptr->info.x << " " << ptr->info.y << " " << ptr->info.z << std::endl;
	else if (level > 1)
	{
		givenlevel(ptr->left, level - 1, solution);
		givenlevel(ptr->right, level - 1, solution);
	}

}


template<typename T>
void BST<T>::levelorder(node<T> *ptr, std::ofstream& solution) {

	int h = BST::get_max_depth();
	for (int i = h; i >= 1; i--) {
		BST::givenlevel(ptr, i, solution);
	}

}


template<typename T>
void BST<T>::display(node<T> *ptr, int level) {
	int i;
	if (ptr != NULL)
	{
		display(ptr->right, level + 1);
		std::cout << std::endl;
		if (ptr == root) {
			std::cout << "Root->: ";
		}
		else {
			for (i = 0; i < level; i++) {
				std::cout << "     ";
			}
		}
		std::cout << "(" << ptr->info.x << ", " << ptr->info.y << ", " << ptr->info.z << ")   ";
		display(ptr->left, level + 1);
	}
}



template<typename T>
T Jacobian_cost(const std::vector<Triplet<T>>& Jac,
	std::vector<std::vector<Cost_Triplet<T>>>& CJ) {

	T p = Jac.size();

	for (unsigned int j = 0; j < p; j++) {
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
T COST_T1(const std::vector<std::vector<Cost_Triplet<T>>>& CH,
	const std::vector<std::vector<Cost_Triplet<T>>>& CJ,
	const int j, const int k, const int i) {

	// Extract the number of slices out of the Hessian tensor.
	// Calculate the cost for symbolic multiplication of Jacobian 
	// chain from j->k and the Hessian chain from (k-1)->i
	T fma = CJ[j][k].dim.x * CH[k - 1][i].dim.x *
		CH[k - 1][i].dim.y * CH[k - 1][i].dim.z;

	T cost_1 = CJ[j][k].cost + CH[k - 1][i].cost + fma;


	return cost_1;
}



template<typename T>
T COST_T2(const std::vector<std::vector<Cost_Triplet<T>>>& CH,
	const std::vector<std::vector<Cost_Triplet<T>>>& CJ,
	const int j, const int k, const int i) {


	T cost_2 = CH[j][k].cost + CJ[k - 1][i].cost +
		CH[j][k].dim.x * CJ[k - 1][i].dim.x *
		CJ[k - 1][i].dim.y * (CH[j][k].dim.y +
			CJ[k - 1][i].dim.y);

	return cost_2;
}


template<typename T>
void Hessian_cost(const std::vector<Triplet<T>>& Hess,
	const std::vector<std::vector<Cost_Triplet<T>>>& CJ,
	std::vector<std::vector<Cost_Triplet<T>>>& CH) {

	T p = Hess.size();

	for (unsigned int j = 0; j < p; j++) {
		for (int i = j; i >= 0; i--) {
			if (i == j) {
				CH[j][i].cost = 0;
				CH[j][i].split_pos = 0;
				CH[j][i].dim = Hess[j];
			}
			else {
				for (int k = i + 1; k <= j; k++) {

					T cost_1 = COST_T1(CH, CJ, j, k, i);
					T cost_2 = COST_T2(CH, CJ, j, k, i);

					T total_cost = cost_1 + cost_2;

					if (k == i + 1 || total_cost < CH[j][i].cost) {
						CH[j][i].cost = total_cost;
						CH[j][i].split_pos = k;
					}

				}
				CH[j][i].dim.x = CJ[j][CH[j][i].split_pos].dim.x;
				CH[j][i].dim.y = CH[CH[j][i].split_pos - 1][i].dim.y;
				CH[j][i].dim.z = CH[CH[j][i].split_pos - 1][i].dim.z;
			}
		}
	}

}



template<typename T>
std::pair<T, T> Random_Bracket(const std::vector<Triplet<T>>& Jac,
	const std::vector<Triplet<T>>& Hess) {

	T p = Hess.size();
	std::vector<std::vector<Cost_Triplet<T>>>
		CHB(p, std::vector<Cost_Triplet<T>>(p));


	// Bracketing from right (A(B(C(D.....(X)))))
	CHB[0][0].cost = 0;
	CHB[0][0].dim = Hess[0];
	for (unsigned int j = 1; j < p; j++) {

		T cost_1_right = Jac[j].x * CHB[j - 1][0].dim.x *
			CHB[j - 1][0].dim.y * CHB[j - 1][0].dim.z +
			CHB[j - 1][0].cost;

		T cost_2_right = 0;
		for (int r = 1; r < j; r++) {
			cost_2_right = cost_2_right + Jac[r].x * Jac[r].y *
				Jac[0].y;
		}
		cost_2_right = cost_2_right + Hess[j].x * Hess[j].y *
			Hess[j].z * Jac[0].y;
		cost_2_right = cost_2_right + Hess[j].x * Hess[j].y *
			Jac[0].y * Jac[0].y;


		T total_cost_right = cost_1_right + cost_2_right;

		// Store for use in following iterations
		CHB[j][0].cost = total_cost_right;
		CHB[j][0].dim.x = Jac[j].x;
		CHB[j][0].dim.y = CHB[j - 1][0].dim.y;
		CHB[j][0].dim.z = CHB[j - 1][0].dim.z;


	}
	T fma_right = CHB[p - 1][0].cost;



	// Bracketing from left ((((A)B)C)D.....X)
	CHB[p - 1][p - 1].cost = 0;
	CHB[p - 1][p - 1].dim = Hess[p - 1];
	for (int j = p - 1; j > 0; j--) {

		T cost_1_left = 0;
		for (int r = p - 1; r > j; r--) {
			cost_1_left = cost_1_left + Jac[p - 1].x * Jac[r - 1].x *
				Jac[r - 1].y;
		}
		cost_1_left = cost_1_left + Jac[p - 1].x * Hess[j - 1].x *
			Hess[j - 1].y * Hess[j - 1].z;


		T cost_2_left = CHB[p - 1][j].cost + CHB[p - 1][j].dim.x *
			Jac[j - 1].x * Jac[j - 1].y * (CHB[p - 1][j].dim.y +
				Jac[j - 1].y);

		T total_cost_left = cost_1_left + cost_2_left;

		// Store for use in following iterations
		CHB[p - 1][j - 1].cost = total_cost_left;
		CHB[p - 1][j - 1].dim.x = Jac[p - 1].x;
		CHB[p - 1][j - 1].dim.y = Hess[j - 1].y;
		CHB[p - 1][j - 1].dim.z = Hess[j - 1].z;


	}
	T fma_left = CHB[p - 1][0].cost;


	CHB.clear();

	return std::make_pair(fma_left, fma_right);

}



template<typename T>
void Bracketing_structure(const std::vector<std::vector<Cost_Triplet<T>>>& C,
	T n, BST<T> &bst, node<T> *root) {

	T start = 0;
	T end = 0;

	std::vector<Triplet<T>> Bin_vector;
	std::vector<T> diff;

	bst.inorder(root, Bin_vector);


	for (unsigned int i = 0; i <= Bin_vector.size(); i++) {

		if (i == 0)  start = n - 1;
		else start = end - 1;

		if (i == Bin_vector.size()) end = 0;
		else end = Bin_vector[i].y;

		node<T> *k;
		k = new node<T>;
		k->info.x = start;
		k->info.y = C[start][end].split_pos;
		k->info.z = end;


		if (k->info.y != 0) { bst.insert(root, k); }

		diff.push_back(start - end);

	}

	if (std::all_of(diff.begin(), diff.end(), [&](int p)
	{return p == 0; })) {
		return;
	}
	else { Bracketing_structure(C, n, bst, root); }
}




int main(int argc, char* argv[]) {

	assert(argc == 2); std::ifstream in(argv[1]);
	using T = unsigned long;


	T p; in >> p; assert(p > 0);              // This gives us the order of compositeness of the hessian 

	std::vector<Triplet<T>> Jac(p);           // Jacobian vector
	std::vector<Triplet<T>> Hess(p);          // Hessian vector

	// Costs for Jacobain and Hessian vector
	std::vector<std::vector<Cost_Triplet<T>>> CJ(p, std::vector<Cost_Triplet<T>>(p));
	std::vector<std::vector<Cost_Triplet<T>>> CH(p, std::vector<Cost_Triplet<T>>(p));


	// Input vector 
	for (unsigned int i = 0; i < p; i++) {
		in >> Jac[i].x >> Jac[i].y;
		Hess[i].x = Jac[i].x;
        Hess[i].y = Jac[i].y;
        Hess[i].z = Jac[i].y;
		Jac[i].z = 1;		
	}


	// Brute force way of computing the FMA for the Hessian chain 
	// (Bracket from right and left reusing the subproblem results)
	std::pair<T, T> Bracket_cost = Random_Bracket(Jac, Hess);
	T left_cost = Bracket_cost.first;
	T right_cost = Bracket_cost.second;


	// Dynamic Programming Algorithm 
	Jacobian_cost(Jac, CJ);                                   
	Hessian_cost(Hess, CJ, CH);
	
	
	// Write the results
	std::cout << "left bracketing fma = " << left_cost << std::endl;
	std::cout << "right bracketing fma = " << right_cost << std::endl;
	std::cout << "optimized bracketing fma = " << CH[p - 1][0].cost << std::endl;
	std::cout << std::endl;
	std::cout << "Dynamic Programming Table:" << std::endl;
	for (unsigned int j = 0; j < p; j++) {
		for (int i = j - 1; i >= 0; i--) {
			std::cout << "fma(F''(" << j << "," << i << "))="
				<< CH[j][i].cost << "; "
				<< "Split before " << CH[j][i].split_pos << "; "
				<< "dim(F''(" << j << "," << i << "))="
				<< CH[j][i].dim.x << "x" << CH[j][i].dim.y
				<< "x" << CH[j][i].dim.z << std::endl;
		}
	}


	// Create the bracketing structure for run time comparison of actual Hessian computation
	T size = p;
	std::ofstream solution;
	solution.open("solution.txt");
	BST<T> bst_CH;
	node<T> *temp_CH;
	temp_CH = new node<T>;
	temp_CH->info.x = size - 1;
	temp_CH->info.y = CH[size - 1][0].split_pos;
	temp_CH->info.z = 0;
	node<T> *root = NULL;
	root = bst_CH.insert(root, temp_CH);
	Bracketing_structure(CH, size, bst_CH, root);
	bst_CH.levelorder(root, solution);
	bst_CH.clear();
	solution.close();

	return 0;
}
