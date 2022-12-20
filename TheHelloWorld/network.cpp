#include "network.h"

neural_network* create_neural_network(int num_hidden_layers, int num_nodes_per_hidden_layer)
{
	neural_network* nn = (neural_network*)malloc(sizeof(neural_network));
	if (nn == nullptr)
	{
		return nullptr;
	}
	
	nn->hidden_nodes = (node**)malloc(sizeof(node*) * num_hidden_layers);
	
	for (int i = 0; i < num_hidden_layers; i++)
	{
		nn->hidden_nodes[i] = (node*)malloc(sizeof(node) * num_nodes_per_hidden_layer);
	}
	
	nn->output_nodes = (output_node*)malloc(sizeof(output_node) * 9);
	
	for (int i = 0; i < INPUT_PIC_DIMENSION_X * INPUT_PIC_DIMENSION_Y; i++)
	{
		nn->input_nodes[i] = *create_node(9);
	}
	
	return nn;
}
