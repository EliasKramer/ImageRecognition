#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include "training_data.hpp"

typedef struct {
	//the index is the sum all previous layers nodes
	//this helps giving each element in a 1d array an index
	int* index_helper;
	float* weights;
	float* biases;
} nn_state_t;

typedef struct {
	int num_layers;
	int* layer_sizes;

	float* activations;
	
	nn_state_t state;
	
	std::string* output_labels;
} n_network_t;


n_network_t* create_network(
	int input_size, 
	std::vector<int>& hidden_layer_sizes, 
	std::vector<std::string>& output_labels);
void delete_network(n_network_t* network);
/*
void set_input(n_network_t& network, const digit_image_t& training_data);
void feed_forward(n_network_t& network);
void apply_noise(n_network_t& network, float noise_range);
void print_output_data(n_network_t& network);
std::string get_output_label(n_network_t& network);
float get_cost(n_network_t& network);

float test_nn(n_network_t& network, const digit_image_collection_t& training_data_collection);
float test_nn_with_printing(n_network_t& network, const digit_image_collection_t& training_data_collection);

void train_on_images(n_network_t& network, const digit_image_collection_t& training_data_collection, int num_epochs, int batch_size);

bool saved_network_exists(std::string file_path);
void save_network(n_network_t& network, std::string file_path);
n_network_t* load_network(std::string file_path);

void print_weights(n_network_t& network);
void print_biases(n_network_t& network);	*/