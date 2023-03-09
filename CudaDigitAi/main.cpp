#include <iostream>
#include "neural_network.hpp"
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;

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

	std::cout << std::endl << "Creating network..." << std::endl << std::endl;

	n_network_t *network;

	if (saved_network_exists("network"))
	{
		std::cout << "loading existing network..." << std::endl;
		network = load_network("network");
	}
	else {
		std::cout << "generating new network" << std::endl;
		network = create_network(28 * 28, 2, 25, 10);
		std::cout << "applying noise " << std::endl;
		apply_noise(*network, 0.1f);
	}

	float current_best = test_nn(*network, testing_data_mnist);

	for (int i = 0; i < 1000; i++)
	{
		train_on_images(*network, training_data_mnist, 1000, 60);
		float current = test_nn_with_printing(*network, testing_data_mnist);
		if (current > current_best)
		{
			std::cout << "saving network..." << std::endl;
			save_network(*network, "network");
			current_best = current;
		}
		std::cout << "current best is " << current_best << std::endl;
	}

	delete_network(network);
	return 0;
}