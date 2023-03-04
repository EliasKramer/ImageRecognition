#include <iostream>
#include "neural_network.h"
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	n_network_t& network = create_network(28 * 28, 2, 16, 10);

	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	std::cout << std::endl << "Reading files..." << std::endl;
	digit_image_collection_t training_data_mnist = load_mnist_data(
		"/data/train-images.idx3-ubyte",
		"/data/train-labels.idx1-ubyte");

	digit_image_collection_t testing_data_mnist = load_mnist_data(
		"/data/t10k-images.idx3-ubyte",
		"/data/t10k-labels.idx1-ubyte");

	const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
	std::cout 
		<< "Reading files done. took : " 
		<< std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

	apply_noise(network, 0.1f);
	
	std::cout << std::endl << "Testing network..." << std::endl;
	float percent_correct = test_nn(network, testing_data_mnist);
	std::cout << std::endl << "Testing done. Percent correct: " << percent_correct << "%" << std::endl;

	/*
	int count = 0;
	for (digit_image& image : training_data_mnist)
	{
		print_digit_image(image);
		if (count > 5)
		{
			break;
		}
		count++;
	}*/

	delete_network(network);
	return 0;
}