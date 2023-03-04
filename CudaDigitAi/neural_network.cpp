#include "neural_network.hpp"

// private functions
inline float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

inline float cost(float actual, float expected)
{
	return (actual - expected) * (actual - expected);
}

inline float sigmoid_derivative(float x)
{
	float sigmoid_activation = sigmoid(x);
	return sigmoid_activation * (1.0f - sigmoid_activation);
}

inline float cost_derivative(float actual, float expected)
{
	return 2.0f * (actual - expected);
}

inline float rand_between(float min, float max)
{
	// Get the current time as a seed
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

	// Create a random engine and distribution
	std::default_random_engine engine(seed);
	std::uniform_real_distribution<float> distribution(min, max);

	// Generate a random number within the given range
	return distribution(engine);
}

void clear_state(nn_state_t& state)
{
	for (int i = 0; i < state.num_layers - 1; i++)
	{
		for (int j = 0; j < state.layer_sizes[i]; j++)
		{
			for (int k = 0; k < state.layer_sizes[i + 1]; k++)
			{
				state.weights[i][j][k] = 0.0f;
			}
		}
	}

	for (int i = 0; i < state.num_layers - 1; i++)
	{
		for (int j = 0; j < state.layer_sizes[i + 1]; j++)
		{
			state.biases[i][j] = 0.0f;
		}
	}
}
//TODO write tests for this function
nn_state_t& get_empty_state(n_network_t& network)
{
	nn_state_t* state = new nn_state_t;

	state->num_layers = network.num_layers;
	state->layer_sizes = new int[state->num_layers];
	for (int i = 0; i < state->num_layers; i++)
	{
		state->layer_sizes[i] = network.layer_sizes[i];
	}

	state->weights = new float** [state->num_layers - 1];
	for (int i = 0; i < state->num_layers - 1; i++)
	{
		state->weights[i] = new float* [state->layer_sizes[i]];
		for (int j = 0; j < state->layer_sizes[i]; j++)
		{
			state->weights[i][j] = new float[state->layer_sizes[i + 1]];
		}
	}

	state->biases = new float* [state->num_layers - 1];
	for (int i = 0; i < state->num_layers - 1; i++)
	{
		state->biases[i] = new float[state->layer_sizes[i + 1]];
	}

	clear_state(*state);

	return *state;
}

//TODO write tests for this function
void delete_state(nn_state_t& state)
{
	for (int i = 0; i < state.num_layers - 1; i++)
	{
		for (int j = 0; j < state.layer_sizes[i]; j++)
		{
			delete[] state.weights[i][j];
		}
		delete[] state.weights[i];
	}
	delete[] state.weights;

	for (int i = 0; i < state.num_layers - 1; i++)
	{
		delete[] state.biases[i];
	}
	delete[] state.biases;
	
	delete[] state.layer_sizes;
}

void init_network(n_network_t& network)
{
	//set activations to 0
	for (int i = 0; i < network.num_layers; i++)
	{
		for (int j = 0; j < network.layer_sizes[i]; j++)
		{
			network.activations[i][j] = 0.0f;
		}
	}

	//set weights to 0
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		for (int j = 0; j < network.layer_sizes[i]; j++)
		{
			for (int k = 0; k < network.layer_sizes[i + 1]; k++)
			{
				network.weights[i][j][k] = 0.0f;
			}
		}
	}

	//set biases to 0
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		for (int j = 0; j < network.layer_sizes[i + 1]; j++)
		{
			network.biases[i][j] = 0.0f;
		}
	}
}

//public functions

n_network_t& create_network(
	int input_size,
	int num_of_hidden_layers,
	int hidden_layer_size,
	int num_of_output_layer)
{
	n_network_t* network = new n_network_t;

	//size initialization and allocation
	network->num_layers = num_of_hidden_layers + 2;
	network->layer_sizes = new int[network->num_layers];

	network->layer_sizes[0] = input_size;
	for (int i = 1; i <= num_of_hidden_layers; i++)
	{
		network->layer_sizes[i] = hidden_layer_size;
	}
	network->layer_sizes[network->num_layers - 1] = num_of_output_layer;

	//activation initialization and allocation
	network->activations = new float* [network->num_layers];
	for (int i = 0; i < network->num_layers; i++)
	{
		network->activations[i] = new float[network->layer_sizes[i]];
	}

	//weights and biases initialization and allocation
	network->weights = new float** [network->num_layers - 1];
	network->biases = new float* [network->num_layers - 1];

	for (int i = 0; i < network->num_layers - 1; i++)
	{
		//left side is i
		network->weights[i] = new float* [network->layer_sizes[i]];
		network->biases[i] = new float[network->layer_sizes[i + 1]];

		//iterate over the current layer nodes 
		//and allocate a new array of weights for each node on the next layer
		for (int j = 0; j < network->layer_sizes[i]; j++)
		{
			//right side is j
			network->weights[i][j] = new float[network->layer_sizes[i + 1]];
		}
	}

	//init labels
	network->labels = new std::string[network->layer_sizes[network->num_layers - 1]];
	for (int i = 0; i < network->layer_sizes[network->num_layers - 1]; i++)
	{
		network->labels[i] = std::to_string(i);
	}

	init_network(*network);
	
	return *network;
}

void delete_network(n_network_t& network)
{
	//delete activations
	for (int i = 0; i < network.num_layers; i++)
	{
		delete[] network.activations[i];
	}
	delete[] network.activations;

	//iterate over all layers
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		//iterate over each node
		for (int j = 0; j < network.layer_sizes[i]; j++)
		{
			//delete array for current node, 
			//that has an array to the next layer nodes
			delete[] network.weights[i][j];
		}
		//delete current layer of weights
		delete[] network.weights[i];

		//delete current layer of biases
		delete[] network.biases[i];
	}

	//delete weights and biases arrays
	delete[] network.weights;
	delete[] network.biases;

	//delete layer sizes
	delete[] network.layer_sizes;

	//delete labels
	delete[] network.labels;
}

void set_input(n_network_t& network, const digit_image_t& training_data)
{
	//can be removed for more performance

	if (network.layer_sizes[0] != (IMAGE_SIZE_X * IMAGE_SIZE_Y))
	{
		std::cerr << "Error: input size does not match network input size";
		exit(1);
	}

	for (int y = 0; y < IMAGE_SIZE_Y; y++)
	{
		for (int x = 0; x < IMAGE_SIZE_X; x++)
		{
			int idx = y * IMAGE_SIZE_X + x;
			network.activations[0][idx] = training_data.matrix[x][y];
		}
	}
}

void feed_forward(n_network_t& network)
{
	//this is the layer of the current activations we are setting
	for (int layer_idx = 1; layer_idx < network.num_layers; layer_idx++)
	{
		//this is the node of the current layer we are setting
		for (int node_idx = 0; node_idx < network.layer_sizes[layer_idx]; node_idx++)
		{
			float sum = 0.0f;
			//iteration over previous nodes
			for (int prev_node_idx = 0; prev_node_idx < network.layer_sizes[layer_idx - 1]; prev_node_idx++)
			{
				sum += network.activations[layer_idx - 1][prev_node_idx] *
					network.weights[layer_idx - 1][prev_node_idx][node_idx];
			}
			sum += network.biases[layer_idx - 1][node_idx];
			network.activations[layer_idx][node_idx] = sigmoid(sum);
		}
	}
}

void apply_noise(n_network_t& network, float noise_range)
{
	//for weights
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		for (int j = 0; j < network.layer_sizes[i]; j++)
		{
			for (int k = 0; k < network.layer_sizes[i + 1]; k++)
			{
				network.weights[i][j][k] += rand_between(-noise_range, noise_range);
			}
		}
	}
	//for biases
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		for (int j = 0; j < network.layer_sizes[i + 1]; j++)
		{
			network.biases[i][j] += rand_between(-noise_range, noise_range);
		}
	}
}

void print_output_data(n_network_t& network)
{
	std::cout << std::endl << "-----------------------------" << std::endl;
	for (int i = 0; i < network.layer_sizes[network.num_layers-1]; i++)
	{
		float activation = network.activations[network.num_layers - 1][i];
		std::cout << i << " value: " << activation << "\n";
	}
	std::cout << "-----------------------------" << std::endl;
}

std::string get_output_label(n_network_t& network)
{
	int output_idx = network.num_layers - 1;
	//curr float is min float
	float curr_max = std::numeric_limits<float>::min();
	int curr_max_idx = 0;
	for (int i = 0; i < network.layer_sizes[output_idx]; i++)
	{
		if (network.activations[output_idx][i] > curr_max)
		{
			curr_max = network.activations[output_idx][i];
			curr_max_idx = i;
		}
	}
	return std::to_string(curr_max_idx);
}

float get_cost(n_network_t& network)
{
	return 0.0f;
}

float test_nn(n_network_t& network, const digit_image_collection_t& training_data_collection)
{
	//returns the percentage of correct answers
	int correct_answers = 0;
	for (const digit_image_t& curr : training_data_collection)
	{
		set_input(network, curr);
		feed_forward(network);
		std::string output_label = get_output_label(network);
		if (output_label == curr.label)
		{
			correct_answers++;
		}
	}
	return (float)correct_answers / (float)training_data_collection.size() * 100;
}
void test_nn_with_printing(n_network_t& network, const digit_image_collection_t& training_data_collection)
{
	std::cout << std::endl << "Testing network..." << std::endl;
	float percent_correct = test_nn(network, training_data_collection);
	std::cout << std::endl << "Testing done. Percent correct: " << percent_correct << "%" << std::endl;
}
void backprop(
	n_network_t& network, 
	int current_layer_idx, 
	float* unhappiness_before, 
	int unhappiness_before_size,
	nn_state_t& state)
{
	if (current_layer_idx == 0)
	{
		return;
	}

	//the unhappiness for the next layer is always as big as the current layer
	float* unhappiness_for_next_layer = new float[network.layer_sizes[current_layer_idx]];
	//the index of the layer that will be calculated after the current one
	int prev_layer_idx = current_layer_idx - 1;

	for (int i = 0; i < network.layer_sizes[current_layer_idx]; i++)
	{
		//the current activation
		float activation = network.activations[current_layer_idx][i];
		//the bias of the current node
		float bias = network.biases[current_layer_idx-1][i];

		float input_without_activation_function = 0.0f;
		for (int j = 0; j < network.layer_sizes[prev_layer_idx]; j++)
		{
			input_without_activation_function += 
				network.activations[prev_layer_idx][j] * 
				network.weights[prev_layer_idx][j][i];
		}
		input_without_activation_function += bias;

		float d_sigmoid = sigmoid_derivative(input_without_activation_function);

		float weighted_unhappiness = 0.0f;
		for (int j = 0; j < unhappiness_before_size; j++)
		{
			weighted_unhappiness += unhappiness_before[j] * network.weights[current_layer_idx][i][j];
		}

		for (int j = 0; j < network.layer_sizes[prev_layer_idx]; j++)
		{
			float desired_change = weighted_unhappiness * d_sigmoid * network.activations[prev_layer_idx][j];
		}
		float desired_change_bias = weighted_unhappiness * d_sigmoid;
		
		unhappiness_for_next_layer[i] = weighted_unhappiness * d_sigmoid * activation;
	}
	backprop(network, current_layer_idx - 1, unhappiness_for_next_layer, network.layer_sizes[current_layer_idx], state);

	delete[] unhappiness_for_next_layer;
}

void train_on_images(n_network_t& network, const digit_image_collection_t& training_data_collection, int num_epochs)
{
	int output_idx = network.num_layers - 1;

	nn_state_t& desired_changes = get_empty_state(network);
	
	for each (const digit_image_t& curr in training_data_collection)
	{
		set_input(network, curr);
		feed_forward(network);

		//could do this outside of the loop
		float* unhappiness_for_next_layer = new float[network.layer_sizes[output_idx]];

		for (int i = 0; i < network.layer_sizes[output_idx]; i++)
		{
			//current output node activation
			float activation = network.activations[output_idx][i];
			//the expected value for the current output node
			float expected = 0.0f;
			if (curr.label == std::to_string(i))
			{
				expected = 1.0f;
			}
			//the bias of the current output node 
			float bias = network.biases[output_idx-1][i];
			//recreating the input to the current output node without the activation function like sigmoid
			float input_without_activation_function = 0.0f;
			//this segment could be improved, by saving it on the feed forward process
			for (int j = 0; j < network.layer_sizes[network.num_layers - 2]; j++)
			{
				input_without_activation_function += 
					network.weights[network.num_layers - 2][j][i] * 
					network.activations[network.num_layers - 2][j];
			}
			input_without_activation_function += bias;

			//cost derivative
			float unhappiness = cost_derivative(activation, expected);
			//activation function (sigmoid) derivative
			float d_sigmoid = sigmoid_derivative(input_without_activation_function);

			//add model to save this - im not sure about the
			for (int j = 0; j < network.layer_sizes[network.num_layers - 2]; j++)
			{
				//this is how much the weight should change
				float desired_weight_change = unhappiness * d_sigmoid * network.activations[network.num_layers - 2][j] * -1;
			}
			//this is how much the bias should change
			float desired_bias_change = unhappiness * d_sigmoid * -1;

			unhappiness_for_next_layer[i] = unhappiness * d_sigmoid;
		}
		backprop(network, output_idx - 1, unhappiness_for_next_layer, network.layer_sizes[output_idx], desired_changes);

		delete[] unhappiness_for_next_layer;

		//print size of network
		std::cout << "size of network: " << network.num_layers << " : " << std::endl;
		for (int i = 0; i < network.num_layers; i++)
		{
			std::cout << network.layer_sizes[i] << " ";
		}
		//size of state
		std::cout << "size of state: " << desired_changes.num_layers << " : " << std::endl;
		for (int i = 0; i < desired_changes.num_layers; i++)
		{
			std::cout << desired_changes.layer_sizes[i] << " ";
		}

		//apply state
		for (int i = 0; i < network.num_layers - 1; i++)
		{
			for (int j = 0; j < network.layer_sizes[i]; j++)
			{
				for (int k = 0; k < network.layer_sizes[i + 1]; k++)
				{ 
					//network.weights[i][k][j] += desired_changes.weights[i][j][k];
				}
				//network.biases[i][j] += desired_changes.biases[i][j];
			}
		}
		for (int i = 0; i < network.num_layers - 1; i++)
		{
			for (int j = 0; j < network.layer_sizes[i]; j++)
			{
				for (int k = 0; k < network.layer_sizes[i + 1]; k++)
				{
					desired_changes.weights[i][j][k] = 0.0f;
					//network.weights[i][j][k] = 0.0f;
				}
				network.biases[i][j] = 0.0f;
			}
		}

		//the desired changes are set back to 0 after each image
		clear_state(desired_changes);
	}
	delete_state(desired_changes);
}

void print_weights(n_network_t& network)
{
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		std::cout << "Layer " << i << " weights: " << std::endl;
		for (int j = 0; j < network.layer_sizes[i]; j++)
		{
			for (int k = 0; k < network.layer_sizes[i + 1]; k++)
			{
				std::cout << network.weights[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

void print_biases(n_network_t& network)
{
	for (int i = 0; i < network.num_layers - 1; i++)
	{
		std::cout << "Layer " << i << " biases: " << std::endl;
		for (int j = 0; j < network.layer_sizes[i + 1]; j++)
		{
			std::cout << network.biases[i][j] << " ";
		}
		std::cout << std::endl;
	}
}