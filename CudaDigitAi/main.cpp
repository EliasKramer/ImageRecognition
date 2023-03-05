#include <iostream>
#include "neural_network.hpp"
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

	//apply_noise(network, 0.1f);
	

	//training sublist with only one image

	print_digit_image(training_data_mnist[0]);

	test_nn_with_printing(network, testing_data_mnist);
	print_output_data(network);
	print_biases(network);

	apply_noise(network, 15.0f);
	train_on_images(network, training_data_mnist, 500, 50);
	test_nn_with_printing(network, testing_data_mnist);

	//training on all images
	//print_output_data(network);
	//print_biases(network);

	delete_network(network);
	return 0;
}