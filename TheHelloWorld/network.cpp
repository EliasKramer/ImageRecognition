#include "network.h"

neural_network* create_neural_network(int num_hidden_layers, int num_nodes_per_hidden_layer)
{
	//malloc neural network struct
	neural_network* nn = (neural_network*)malloc(sizeof(neural_network));
	if (nn == nullptr)
	{
		std::cerr << "Error: Could not allocate neural network" << std::endl;
		return nullptr;
	}

	//malloc input nodes
	int num_input_nodes = INPUT_PIC_DIMENSION_X * INPUT_PIC_DIMENSION_Y;
	nn->input_nodes = (input_node*)malloc(sizeof(input_node) * num_input_nodes);
	if (nn->input_nodes == nullptr)
	{
		std::cerr << "Error: Could not allocate memory for input nodes." << std::endl;
		free(nn);
		return nullptr;
	}
	for (int i = 0; i < num_input_nodes; i++)
	{
		nn->input_nodes[i].value = 0;
	}

	
	//malloc the array of hidden layers
	nn->hidden_nodes = (node**)malloc(sizeof(node*) * num_hidden_layers);
	if (nn->hidden_nodes == nullptr)
	{
		std::cerr << "Error: Could not allocate memory for hidden nodes." << std::endl;
		free(nn->input_nodes);
		free(nn);
		return nullptr;
	}
	for (int i = 0; i < num_hidden_layers; i++)
	{
		//allocate each hidden layer
		nn->hidden_nodes[i] = (node*)malloc(sizeof(node) * num_nodes_per_hidden_layer);
		if (nn->hidden_nodes[i] == nullptr)
		{
			std::cerr << "Error: Could not allocate memory for hidden nodes." << std::endl;
			for (int j = 0; j < i; j++)
			{
				free(nn->hidden_nodes[j]);
			}
			free(nn->hidden_nodes);
			free(nn->input_nodes);
			free(nn);
			return nullptr;
		}
		//set each value of the hidden layers
		for (int j = 0; j < num_nodes_per_hidden_layer; j++)
		{
			nn->hidden_nodes[i][j].value = 0;
			nn->hidden_nodes[i][j].bias = 0;
			int curr_weights = (i == 0) ? num_input_nodes : num_nodes_per_hidden_layer;
			
			nn->hidden_nodes[i][j].weights = (float*)malloc(sizeof(float) * curr_weights);
			
			if (nn->hidden_nodes[i][j].weights == nullptr)
			{
				std::cerr << "Error: Could not allocate memory for hidden nodes." << std::endl;
				for (int k = 0; k < j; k++)
				{
					free(nn->hidden_nodes[i][k].weights);
				}
				for (int k = 0; k < i; k++)
				{
					free(nn->hidden_nodes[k]);
				}
				free(nn->hidden_nodes);
				free(nn->input_nodes);
				free(nn);
				return nullptr;
			}
			
			for (int k = 0; k < curr_weights; k++)
			{
				nn->hidden_nodes[i][j].weights[k] = 0;
			}
		}
	}
	//malloc ouput nodes
	nn->output_nodes = (output_node*)malloc(sizeof(output_node) * 9);
	
	if (nn->output_nodes == nullptr)
	{
		std::cerr << "Error: Could not allocate memory for output nodes." << std::endl;
		for (int i = 0; i < num_hidden_layers; i++)
		{
			for (int j = 0; j < num_nodes_per_hidden_layer; j++)
			{
				free(nn->hidden_nodes[i][j].weights);
			}
			free(nn->hidden_nodes[i]);
		}
		free(nn->hidden_nodes);
		free(nn->input_nodes);
		free(nn);
		return nullptr;
	}
	
	for (int i = 0; i < 9; i++)
	{
		nn->output_nodes[i].output_name = "OUTPUT: " + std::to_string(i);
		nn->output_nodes[i].node.bias = 0;
		nn->output_nodes[i].node.value = 0;
		nn->output_nodes[i].node.weights = (float*)malloc(sizeof(float) * num_nodes_per_hidden_layer);
		if (nn->output_nodes[i].node.weights == nullptr)
		{
			std::cerr << "Error: Could not allocate memory for output nodes." << std::endl;
			for (int j = 0; j < i; j++)
			{
				free(nn->output_nodes[j].node.weights);
			}
			for (int j = 0; j < num_hidden_layers; j++)
			{
				for (int k = 0; k < num_nodes_per_hidden_layer; k++)
				{
					free(nn->hidden_nodes[j][k].weights);
				}
				free(nn->hidden_nodes[j]);
			}
			free(nn->hidden_nodes);
			free(nn->input_nodes);
			free(nn);
			return nullptr;
		}
	}
	
	return nn;
}
