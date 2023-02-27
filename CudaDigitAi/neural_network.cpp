#include "neural_network.h"

n_network& create_network(
	int input_size, 
	int num_of_hidden_layers, 
	int hidden_layer_size, 
	int num_of_output_layer)
{
	n_network* network = new n_network;

	network->num_layers = num_of_hidden_layers + 2;
	network->layer_sizes = new int[network->num_layers];

	network->layer_sizes[0] = input_size;
	for (int i = 1; i <= num_of_hidden_layers; i++)
	{
		network->layer_sizes[i] = hidden_layer_size;
	}
	network->layer_sizes[network->num_layers - 1] = num_of_output_layer;

	network->activations = new float*[network->num_layers];
	network->weights = new float**[network->num_layers - 1];
	network->biases = new float*[network->num_layers - 1];

	for (int i = 0; i < network->num_layers; i++)
	{
		network->activations[i] = new float[network->layer_sizes[i]];
	}

	for (int i = 0; i < network->num_layers - 1; i++)
	{
		network->weights[i] = new float*[network->layer_sizes[i]];
		network->biases[i] = new float[network->layer_sizes[i + 1]];

		for (int j = 0; j < network->layer_sizes[i]; j++)
		{
			network->weights[i][j] = new float[network->layer_sizes[i + 1]];
		}
	}

	return *network;
}

void delete_network(n_network& network)
{
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