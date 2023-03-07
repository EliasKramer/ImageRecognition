#include <iostream>
#include "neural_network.hpp"
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	n_network_t& network = create_network(28 * 28, 2, 10, 10);

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

	//apply_noise(network, 0.1f);
	

	//training sublist with only one image

	test_nn_with_printing(network, testing_data_mnist);

	//apply_noise(network, 1.0f);
	print_biases(network);
	train_on_images(network, training_data_mnist, 1200, 80);
	
	test_nn_with_printing(network, testing_data_mnist);
	print_biases(network);
	delete_network(network);
	return 0;
}