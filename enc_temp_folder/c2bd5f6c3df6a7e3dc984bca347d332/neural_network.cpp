#include "neural_network.h"

n_network& create_network(
	int input_size, 
	int num_of_hidden_layers, 
	int hidden_layer_size, 
	int num_of_output_layer)
{
	n_network* network = new n_network;

	//size initialization and allocation
	network->num_layers = num_of_hidden_layers + 2;
	network->layer_sizes = new int[network->num_layers];

	network->layer_sizes[0] = input_size;
	for (int i = 1; i <= num_of_hidden_layers; i++)
	{
		network->layer_sizes[i] = hidden_layer_size;
	}
	network->layer_sizes[network->num_layers - 1] = num_of_output_layer;

	//activation initialization and allocation
	network->activations = new float*[network->num_layers];
	for (int i = 0; i < network->num_layers; i++)
	{
		network->activations[i] = new float[network->layer_sizes[i]];
	}

	//weights and biases initialization and allocation
	network->weights = new float**[network->num_layers - 1];
	network->biases = new float*[network->num_layers - 1];

	for (int i = 0; i < network->num_layers - 1; i++)
	{
		network->weights[i] = new float*[network->layer_sizes[i]];
		network->biases[i] = new float[network->layer_sizes[i + 1]];

		//iterate over the current layer nodes 
		//and allocate a new array of weights for each node on the next layer
		for (int j = 0; j < network->layer_sizes[i]; j++)
		{
			network->weights[i][j] = new float[network->layer_sizes[i + 1]];
		}
	}

	return *network;
}

void delete_network(n_network& network)
{
	//delete activations
	for (int i = 0; i < network.num_layers; i++)
	{
		delete[] network.activations[i];
	}
	delete[] network.activations;

	//iterate over all layers
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		//iterate over each node
		for (int j = 0; j < network.layer_sizes[i]; j++)
		{
			//delete array for current node, 
			//that has an array to the next layer nodes
			delete[] network.weights[i][j];
		}
		//delete current layer of weights
		delete[] network.weights[i];

		//delete current layer of biases
		delete[] network.biases[i];
	}
	delete[] network.biases;

	//throws error here
	delete[] network.biases;
	//delete weights and biases arrays
	delete[] network.weights;

	//delete layer sizes
	delete[] network.layer_sizes;
}

void set_input(n_network& network, training_data& training_data)
{
}

void feed_forward(n_network& network)
{
}

output_data get_output_data(n_network& network)
{
	return output_data();
}

std::string get_output_label(n_network& network)
{
	return std::string();
}

float get_cost(n_network& network)
{
	return 0.0f;
}