#pragma once
#include <string>

typedef struct{
	float bias;
	float* weights;
	float value;
} node;

typedef struct {
	float value;
} input_node;

typedef struct{
	node node;
	std::string output_name;
} output_node;

node* create_node(int num_weights);
void delete_node(node* n);