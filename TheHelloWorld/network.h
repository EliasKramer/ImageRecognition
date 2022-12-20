#pragma once

#include "node.h"
#include "data.h"

typedef struct {
	node input_nodes[INPUT_PIC_DIMENSION_X * INPUT_PIC_DIMENSION_Y];
	node** hidden_nodes;
	node* output_nodes;
} neural_network;

neural_network* create_neural_network(int num_hidden_layers, int num_nodes_per_hidden_layer);