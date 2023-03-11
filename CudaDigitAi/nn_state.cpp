#include "nn_state.hpp"

nn_state_t* get_empty_state(int num_layers, int* layer_sizes)
{
	nn_state_t* state = new nn_state_t();
	//create activation index helper
	state->activation_idx_helper = new int[num_layers];
	state->activation_idx_helper[0] = 0;
	int temp = 0;
	for (int i = 1; i < num_layers; i++)
	{
		temp += layer_sizes[i - 1];
		state->activation_idx_helper[i] = temp;
	}

	//create bias index helper
	state->bias_idx_helper = new int[num_layers - 1];
	state->bias_idx_helper[0] = 0;
	temp = 0;
	for (int i = 1; i < num_layers - 1; i++)
	{
		temp += layer_sizes[i];
		state->bias_idx_helper[i] = temp;
	}

	//sum all nodes
	int total_nodes = 0;
	for (int i = 0; i < num_layers; i++)
	{
		total_nodes += layer_sizes[i];
	}
	state->total_nodes = total_nodes;

	//create biases (first layer has no biases)
	state->total_biases = total_nodes - layer_sizes[0];
	state->biases = new float[state->total_biases];

	//create weights
	int total_weights = 0;
	for (int i = 1; i < num_layers; i++)
	{
		total_weights += layer_sizes[i] * layer_sizes[i - 1];
	}
	state->total_weights = total_weights;
	state->weights = new float[state->total_weights];

	clear_state(*state);

	return state;
}

void clear_state(nn_state_t& state)
{
	for (int i = 0; i < state.total_biases; i++)
	{
		state.biases[i] = 0.0f;
	}
	for (int i = 0; i < state.total_weights; i++)
	{
		state.weights[i] = 0.0f;
	}
}

void delete_nn_state(nn_state_t* state)
{
	delete[] state->activation_idx_helper;
	delete[] state->bias_idx_helper;
	delete[] state->biases;
	delete[] state->weights;
	delete state;
}
