#pragma once

#include "node.h"
#include "data.h"

typedef struct {
	int num_hidden_layers;
	int num_nodes_per_hidden_layer;
	input_node* input_nodes;
	node** hidden_nodes;
	output_node* output_nodes;
} neural_network;

neural_network* create_neural_network(int num_hidden_layers, int num_nodes_per_hidden_layer);
void free_neural_network(neural_network* nn);