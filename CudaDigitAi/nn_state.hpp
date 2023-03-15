#pragma once

typedef struct {
	//the size of the activation array
	int total_nodes;
	//the size of the weights array
	int total_weights;
	//the size of the biases array
	int total_biases;

	//weights and bias array
	float* weights;
	float* biases;

	//these help arrays are the sum of all previous layers nodes
	//this helps giving each element an 1 dimensional array index
	//
	// here an example:
	// a network with 4 layers and 784, 16, 16, 10 nodes
	//+----+----+----+----+
	//|784 | 16 | 16 | 10 |
	//+----+----+----+----+
	//the activation index helper would be:
	//+----+----+----------+--------------+
	//|  0 |784 | 784 + 16 | 784 + 16 + 16|
	//+----+----+----------+--------------+
	//the bias index helper would be:
	//+----+----+----+
	//|  0 | 16 | 32 |
	//+----+----+----+
	//the bias array is one shorter than the activation array
	//because the first layer has no biases

	//you can now calculate the index of a node in the activation array
	//by adding the activation index helper value of the layer
	//example 
	//activation[activation_idx_helper[2] + 5] is the 5th node of the 3rd layer in a 1D array
	int* activation_idx_helper;
	int* bias_idx_helper;
} nn_state_t;

int get_activation_idx(int* activation_idx_helper, int curr_layer, int curr_neuron);
int get_bias_idx(int* bias_idx_helper, int curr_layer, int curr_neuron);
int get_weight_idx(int* activation_idx_helper, int* layer_sizes, int curr_layer, int curr_neuron, int left_neuron);

nn_state_t* get_empty_state(int num_layers, int* layer_sizes);
void clear_state(nn_state_t& state);
void delete_nn_state(nn_state_t* state);