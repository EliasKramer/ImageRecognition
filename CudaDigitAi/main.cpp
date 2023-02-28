#include <iostream>
#include "neural_network.h"
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;
	/*
	n_network& network = create_network(28 * 28, 2, 16, 10);
	//apply_noise(network, 1.0f);
	feed_forward(network);

	print_output_data(network);
	std::cout << "The answer is " << get_output_label(network) << std::endl;

	delete_network(network);*/

	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	std::vector<training_data> training_data_mnist = load_mnist_data(
		"C:/Users/krame/Desktop/_homemadeAI/CudaDigitAi/CudaDigitAi/data/train-images.idx3-ubyte",
		"C:/Users/krame/Desktop/_homemadeAI/CudaDigitAi/CudaDigitAi/data/train-labels.idx1-ubyte");
	const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
	std::cout << "Exec. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

	int max_images = 20;
	int count = 0;
	for (training_data& curr : training_data_mnist)
	{
		print_training_data(curr);
		count++;
		if (count >= max_images)
		{
			break;
		}
	}

	return 0;
}