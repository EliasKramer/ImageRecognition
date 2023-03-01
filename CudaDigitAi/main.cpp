#include <iostream>
#include "neural_network.h"
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	n_network& network = create_network(28 * 28, 2, 16, 10);

	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	digit_image_collection training_data_mnist = load_mnist_data(
		"../../data/train-images.idx3-ubyte",
		"C:/Users/krame/Desktop/_homemadeAI/CudaDigitAi/CudaDigitAi/data/train-labels.idx1-ubyte");

	digit_image_collection testing_data_mnist = load_mnist_data(
		"C:/Users/krame/Desktop/_homemadeAI/CudaDigitAi/CudaDigitAi/data/t10k-images.idx3-ubyte",
		"C:/Users/krame/Desktop/_homemadeAI/CudaDigitAi/CudaDigitAi/data/t10k-labels.idx1-ubyte");

	const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
	std::cout << "Exec. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

	float percent_correct = test_nn(network, testing_data_mnist);
	
	std::cout << std::endl << "percen correct: " << percent_correct << "%" << std::endl;

	delete_network(network);

	return 0;
}