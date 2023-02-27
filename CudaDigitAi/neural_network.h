#pragma once

#include <vector>
#include <string>
typedef struct {
	int num_layers;
	int* layer_sizes;

	//start at idx 0 and go to n
	float** activations;
	//weights start at idx 0 and go to n-1
	//next idx is current node
	//next idx is the next layer node
	float*** weights;
	//biases start at idx 1 and go to n
	float** biases;
} n_network;

typedef struct {
	float** matrix;
	int rows;
	int cols;
	std::string label;
} training_data;

typedef struct {
	float* activations;
	std::string* label;
} output_data;

typedef std::vector<training_data> training_data_set;

n_network& create_network(int input_size, int num_of_hidden_layers, int hidden_layer_size, int num_of_output_layer);
void delete_network(n_network& network);

void set_input(n_network& network, training_data& training_data);
void feed_forward(n_network& network);
output_data get_output_data(n_network& network);
std::string get_output_label(n_network& network);
float get_cost(n_network& network);