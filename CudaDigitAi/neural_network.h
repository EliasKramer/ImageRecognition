#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include "training_data.h"

typedef struct {
	int num_layers;
	int* layer_sizes;

	//start at idx 0 and go to n
	float** activations;
	//weights start at idx 0 and go to n-1
	//assuming the input starts at the input layer and increases to the output layer
	//the left side is the second index of the weight and the right side is the third idx of the weight
	float*** weights;
	//biases start at idx 1 and go to n
	float** biases;
} n_network;

typedef std::vector<training_data> training_data_set;	

n_network& create_network(int input_size, int num_of_hidden_layers, int hidden_layer_size, int num_of_output_layer);
void delete_network(n_network& network);

void set_input(n_network& network, training_data& training_data);
void feed_forward(n_network& network);
void apply_noise(n_network& network, float noise_range);
void print_output_data(n_network& network);
std::string get_output_label(n_network& network);
float get_cost(n_network& network);

void print_weights(n_network& network);
void print_biases(n_network& network);	