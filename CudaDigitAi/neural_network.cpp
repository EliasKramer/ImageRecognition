#include "neural_network.h"

// private functions
inline float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

//public functions

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
	network->activations = new float* [network->num_layers];
	for (int i = 0; i < network->num_layers; i++)
	{
		network->activations[i] = new float[network->layer_sizes[i]];
	}

	//weights and biases initialization and allocation
	network->weights = new float** [network->num_layers - 1];
	network->biases = new float* [network->num_layers - 1];

	for (int i = 0; i < network->num_layers - 1; i++)
	{
		//left side is i
		network->weights[i] = new float* [network->layer_sizes[i]];
		network->biases[i] = new float[network->layer_sizes[i + 1]];

		//iterate over the current layer nodes 
		//and allocate a new array of weights for each node on the next layer
		for (int j = 0; j < network->layer_sizes[i]; j++)
		{
			//right side is j
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

	//delete weights and biases arrays
	delete[] network.weights;
	delete[] network.biases;

	//delete layer sizes
	delete[] network.layer_sizes;
}

void set_input(n_network& network, training_data& training_data)
{
	//can be removed for more performance

	if (network.layer_sizes[0] != (training_data.cols * training_data.rows))
	{
		printf("Error: input size does not match network input size\n");
		throw "Error: input size does not match network input size";
		return;
	}

	for (int y = 0; y < training_data.rows; y++)
	{
		for (int x = 0; x < training_data.cols; x++)
		{
			int idx = y * training_data.cols + x;
			network.activations[0][idx] = training_data.matrix[x][y];
		}
	}
}

void feed_forward(n_network& network)
{
	//this is the layer of the current activations we are setting
	for (int layer_idx = 1; layer_idx < network.num_layers; layer_idx++)
	{
		//this is the node of the current layer we are setting
		for (int node_idx = 0; node_idx < network.layer_sizes[layer_idx]; node_idx++)
		{
			float sum = 0.0f;
			//iteration over previous nodes
			for (int prev_node_idx = 0; prev_node_idx < network.layer_sizes[layer_idx - 1]; prev_node_idx++)
			{
				sum += network.activations[layer_idx - 1][prev_node_idx] *
					network.weights[layer_idx - 1][prev_node_idx][node_idx];
			}
			sum += network.biases[layer_idx - 1][node_idx];
			network.activations[layer_idx][node_idx] = sigmoid(sum);
		}
	}
}

void apply_noise(n_network& network, float noise_range)
{
	for (int i = 0; i < network.num_layers; i++)
	{
		for (int node_idx = 0; node_idx < network.layer_sizes[i]; node_idx++)
		{
			network.activations[i][node_idx] += (rand() % 100) / 100.0f * noise_range * 2 - noise_range;
		}
	}
}

void print_output_data(n_network& network)
{
	
}

std::string get_output_label(n_network& network)
{
	int output_idx = network.num_layers - 1;
	//curr float is min float
	float curr_max = std::numeric_limits<float>::min();
	int curr_max_idx = 0;
	for (int i = 0; i < network.layer_sizes[output_idx]; i++)
	{
		if (network.activations[output_idx][i] > curr_max)
		{
			curr_max = network.activations[output_idx][i];
			curr_max_idx = i;
		}
	}
	return std::to_string(curr_max_idx);
}

float get_cost(n_network& network)
{
	return 0.0f;
}