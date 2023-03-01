#include <iostream>
#include "neural_network.h"
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	n_network& network = create_network(28 * 28, 2, 16, 10);

	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	digit_image_collection training_data_mnist = load_mnist_data(
		"/data/train-images.idx3-ubyte",
		"/data/train-labels.idx1-ubyte");

	digit_image_collection testing_data_mnist = load_mnist_data(
		"/data/t10k-images.idx3-ubyte",
		"/data/t10k-labels.idx1-ubyte");

	const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
	std::cout << "Exec. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

	float percent_correct = test_nn(network, testing_data_mnist);
	
	std::cout << std::endl << "percent correct: " << percent_correct << "%" << std::endl;

	int count = 0;
	for (digit_image& image : training_data_mnist)
	{
		print_digit_image(image);
		if (count > 5)
		{
			break;
		}
		count++;
	}

	delete_network(network);

	return 0;
}