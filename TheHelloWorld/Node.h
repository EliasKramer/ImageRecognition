#pragma once

typedef struct{
	float bias;
	float* weights;
	float value;
} node;
typedef struct{
	node node;
	char* output_name;
} output_node;

node* create_node(int num_weights);
void delete_node(node* n);