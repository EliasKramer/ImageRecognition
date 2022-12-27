#include "network.h"

neural_network* create_neural_network(int num_hidden_layers, int num_nodes_per_hidden_layer)
{
	//malloc neural network struct
	neural_network* nn = new neural_network;
	nn->num_hidden_layers = num_hidden_layers;
	nn->num_nodes_per_hidden_layer = num_nodes_per_hidden_layer;
	/*if (nn == nullptr)
	{
		std::cerr << "Error: Could not allocate neural network" << std::endl;
		return nullptr;
	}*/

	//malloc input nodes
	int num_input_nodes = INPUT_PIC_DIMENSION_X * INPUT_PIC_DIMENSION_Y;
	nn->input_nodes = new node[num_input_nodes];
	/*if (nn->input_nodes == nullptr)
	{
		std::cerr << "Error: Could not allocate memory for input nodes." << std::endl;
		free(nn);
		return nullptr;
	}*/
	for (int i = 0; i < num_input_nodes; i++)
	{
		nn->input_nodes[i].value = 0;
		nn->input_nodes[i].node = nullptr;
	}

	
	//malloc the array of hidden layers
	nn->hidden_nodes = new node*[num_hidden_layers] ;// (node**)malloc(sizeof(node*) * num_hidden_layers);
	/*
	if (nn->hidden_nodes == nullptr)
	{
		std::cerr << "Error: Could not allocate memory for hidden nodes." << std::endl;
		free(nn->input_nodes);
		free(nn);
		return nullptr;
	}*/
	for (int i = 0; i < num_hidden_layers; i++)
	{
		//allocate each hidden layer
		nn->hidden_nodes[i] = new node[num_hidden_layers];// (node*)malloc(sizeof(node) * num_nodes_per_hidden_layer);
		/*
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
		}*/
		//set each value of the hidden layers
		for (int j = 0; j < num_nodes_per_hidden_layer; j++)
		{
			nn->hidden_nodes[i][j].value = 0;
			
			nn->hidden_nodes[i][j].node = new functional_node;
			nn->hidden_nodes[i][j].node->bias= 0;
			int curr_weights = (i == 0) ? num_input_nodes : num_nodes_per_hidden_layer;
			nn->hidden_nodes[i][j].node->weights = new float[curr_weights];// (float*)malloc(sizeof(float) * curr_weights);
			
			/*
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
			*/
			for (int k = 0; k < curr_weights; k++)
			{
				nn->hidden_nodes[i][j].node->weights[k] = 0;
			}
		}
	}
	//malloc ouput nodes
	nn->output_nodes = new output_node[9];// (output_node*)malloc(sizeof(output_node) * 9);
	/*
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
	*/
	for (int i = 0; i < 9; i++)
	{
		std::string output_string = "OUTPUT: " + std::to_string(i);
		nn->output_nodes[i].output_name = output_string;

		nn->output_nodes[i].node.value = 0;
		
		nn->output_nodes[i].node.node = new functional_node;
		nn->output_nodes[i].node.node->bias = 0;
		nn->output_nodes[i].node.node->weights = new float[num_nodes_per_hidden_layer];// (float*)malloc(sizeof(float) * num_nodes_per_hidden_layer);
		/*
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
		}*/
	}
	
	return nn;
}

void free_neural_network(neural_network* nn)
{
	/*// Free output nodes
	for (int i = 0; i < 9; i++)
	{
		delete[] nn->output_nodes[i].node.weights;
	}
	delete[] nn->output_nodes;

	// Free hidden nodes
	// Iterate through each hidden layer
	for (int i = 0; i < nn->num_hidden_layers; i++)
	{
		// Iterate through each node in the hidden layer
		for (int j = 0; j < nn->num_nodes_per_hidden_layer; j++)
		{
			// Iterate through each weight in the node
			delete[] nn->hidden_nodes[i][j].weights;
		}
		// Delete the hidden layer
		delete[] nn->hidden_nodes[i];
	}
	delete[] nn->hidden_nodes;

	// Free input nodes
	delete[] nn->input_nodes;

	// Free neural network
	delete nn;*/
}

void set_input_nodes(input_picture& pic, neural_network& nn)
{
	for (int y = 0; y < INPUT_PIC_DIMENSION_Y; y++)
	{
		for (int x = 0; x < INPUT_PIC_DIMENSION_X; x++)
		{
			nn.input_nodes[y * INPUT_PIC_DIMENSION_X + x].value = pic.pixels[y][x];
		}
	}
}

void process(neural_network& nn)
{
	for (int i = 0; i < nn.num_hidden_layers; i++)
	{
		for (int j = 0; j < nn.num_nodes_per_hidden_layer; j++)
		{
			node* input_nodes = (i == 0) ? nn.input_nodes : nn.hidden_nodes[i - 1];
			int num_input_nodes = (i == 0) ? INPUT_PIC_DIMENSION_X * INPUT_PIC_DIMENSION_Y : nn.num_nodes_per_hidden_layer;
			
			calculate(*nn.hidden_nodes[i], input_nodes, num_input_nodes);
		}
	}
	for (int i = 0; i < 9; i++)
	{
		calculate(nn.output_nodes[i].node, nn.hidden_nodes[nn.num_hidden_layers - 1], nn.num_nodes_per_hidden_layer);
	}
}