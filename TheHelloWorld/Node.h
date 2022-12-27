#pragma once
#include "neural_math.h"
#include <string>

typedef struct{
	float bias;
	float* weights;
} functional_node;

typedef struct {
	float value;
	functional_node* node;
} node;

typedef struct{
	node node;
	std::string output_name;
} output_node;

void calculate(node& n, node* input_nodes, int num_input_nodes);