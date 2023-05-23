#include "neural_network.hpp"
#include <thread>
#include <mutex>

n_network_t* create_network(
	int input_size,
	std::vector<int>& hidden_layer_sizes,
	std::vector<std::string>& output_labels)
{
	n_network_t* network = new n_network_t();

	//create output labels
	network->output_labels = new std::string[output_labels.size()];
	for (int i = 0; i < output_labels.size(); i++)
	{
		network->output_labels[i] = output_labels[i];
	}

	//create size of each layer
	network->num_layers = hidden_layer_sizes.size() + 2;
	network->layer_sizes = new int[network->num_layers];
	network->layer_sizes[0] = input_size;
	for (int i = 0; i < hidden_layer_sizes.size(); i++)
	{
		network->layer_sizes[i + 1] = hidden_layer_sizes[i];
	}
	network->layer_sizes[network->num_layers - 1] = output_labels.size();

	//create weights and biases
	network->state = get_empty_state(network->num_layers, network->layer_sizes);

	//create activations
	network->activations = new float[network->state->total_nodes];

	//set activations to 0
	for (int i = 0; i < network->state->total_nodes; i++)
	{
		network->activations[i] = 0.0f;
	}

	return network;
}

void delete_network(n_network_t* network)
{
	delete[] network->output_labels;
	delete[] network->layer_sizes;
	delete[] network->activations;
	delete_nn_state(network->state);
	delete network;
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
			network.activations[get_activation_idx(network.state->activation_idx_helper, 0, idx)] =
				training_data.matrix[x][y];
		}
	}
}

void feed_forward(n_network_t& network)
{
	for (int curr_layer = 1; curr_layer < network.num_layers; curr_layer++)
	{
		for (int curr_neuron = 0; curr_neuron < network.layer_sizes[curr_layer]; curr_neuron++)
		{
			float sum = 0;
			for (int left_neuron = 0; left_neuron < network.layer_sizes[curr_layer - 1]; left_neuron++)
			{
				sum +=
					network.activations[
						get_activation_idx(
							network.state->activation_idx_helper,
							curr_layer - 1,
							left_neuron)] *
					network.state->weights[
						get_weight_idx(
							network.state->activation_idx_helper,
							network.layer_sizes,
							curr_layer,
							curr_neuron,
							left_neuron)];
			}
			sum += network.state->biases[
				get_bias_idx(network.state->bias_idx_helper, curr_layer, curr_neuron)];
			network.activations[
				get_activation_idx(network.state->activation_idx_helper, curr_layer, curr_neuron)] = sigmoid(sum);
		}
	}
}

void apply_noise(n_network_t& network, float noise_range)
{
	for (int i = 0; i < network.state->total_weights; i++)
	{
		network.state->weights[i] += rand_between(-noise_range, noise_range);
	}
	for (int i = 0; i < network.state->total_biases; i++)
	{
		network.state->biases[i] += rand_between(-noise_range, noise_range);
	}
}

void print_output_data(n_network_t& network)
{
	for (int i = 0; i < network.layer_sizes[network.num_layers - 1]; i++)
	{
		std::cout
			<< network.output_labels[i] << ": "
			<< network.activations[get_activation_idx(network.state->activation_idx_helper, network.num_layers - 1, i)]
			<< std::endl;
	}
}

std::string get_output_label(n_network_t& network)
{
	float max = 0;
	int max_idx = 0;
	for (int i = 0; i < network.layer_sizes[network.num_layers - 1]; i++)
	{
		float curr = network.activations[get_activation_idx(network.state->activation_idx_helper, network.num_layers - 1, i)];
		if (curr > max)
		{
			max = curr;
			max_idx = i;
		}
	}
	return network.output_labels[max_idx];

}

float test_nn(n_network_t& network, const digit_image_collection_t& training_data_collection)
{
	float correct = 0;
	for (const digit_image_t& curr : training_data_collection)
	{
		set_input(network, curr),
			feed_forward(network);

		if (get_output_label(network) == curr.label)
		{
			correct++;
		}
	}

	return correct / training_data_collection.size() * 100.0f;
}

float test_nn_with_printing(n_network_t& network, const digit_image_collection_t& training_data_collection)
{
	std::cout << std::endl << "Testing network..." << std::endl;
	float percent_correct = test_nn(network, training_data_collection);
	std::cout << std::endl << "Testing done. Percent correct: " << percent_correct << "%" << std::endl;
	return percent_correct;
}

void backprop(
	n_network_t& network,
	int current_layer_idx,
	float* unhappiness_right,
	int unhappiness_right_size,
	std::vector<nn_state_t*>& desired_changes,
	int current_image_idx)
{
	if (current_layer_idx == 0)
	{
		return;
	}

	//the unhappiness for the next layer is always as big as the current layer
	float* unhappiness_for_next_layer = new float[network.layer_sizes[current_layer_idx]];
	//the index of the layer that will be calculated after the current one
	int left_idx = current_layer_idx - 1;
	int right_layer_idx = current_layer_idx + 1;
	for (int i = 0; i < network.layer_sizes[current_layer_idx]; i++)
	{
		//the current activation
		int activation_idx = get_activation_idx(network.state->activation_idx_helper, current_layer_idx, i);
		float activation = network.activations[activation_idx];
		//the bias of the current node
		int bias_idx = get_bias_idx(network.state->bias_idx_helper, current_layer_idx, i);
		float bias = network.state->biases[bias_idx];

		float input_without_activation_function = logit(activation);
		float d_sigmoid = sigmoid_derivative(input_without_activation_function);

		float weighted_unhappiness = 0.0f;
		for (int j = 0; j < unhappiness_right_size; j++)
		{
			weighted_unhappiness += unhappiness_right[j] *
				network.state->weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, right_layer_idx, j, i)];
		}

		for (int j = 0; j < network.layer_sizes[left_idx]; j++)
		{
			float desired_change =
				weighted_unhappiness *
				d_sigmoid *
				//network.activations[left_idx][j];
				network.activations[get_activation_idx(network.state->activation_idx_helper, left_idx, j)];
			//set_weight(desired_changes[current_image_idx], current_layer_idx, i, j, desired_change);
			desired_changes[current_image_idx]->
				weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, current_layer_idx, i, j)]
				+= desired_change;
		}
		float desired_change_bias = weighted_unhappiness * d_sigmoid;
		//set_bias(desired_changes[current_image_idx], current_layer_idx, i, desired_change_bias);
		desired_changes[current_image_idx]->biases[bias_idx] += desired_change_bias;

		unhappiness_for_next_layer[i] = weighted_unhappiness * d_sigmoid * activation;
	}
	backprop(network, current_layer_idx - 1, unhappiness_for_next_layer, network.layer_sizes[current_layer_idx], desired_changes, current_image_idx);

	delete[] unhappiness_for_next_layer;
}

static void print_progress(int current, int total, long long elapsed, long long remaining)
{
	float percent = (float)current / (float)total * 100.0f;

	//print percent with 2 decimal places
	std::cout
		<< "Progress: " << std::fixed << std::setprecision(2) << percent << "%"
		<< " | remaining " << ms_to_string(remaining) 
		<< " | elapsed " << ms_to_string(elapsed)
		<< std::flush << std::endl;
}

static void progress_thread(int* current, int total)
{

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point last_time_updated = start;
	do
	{
		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_updated).count() < 1000 && *current < total)
		{
			continue;
		}
		last_time_updated = now;

		float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
		float time_remaining = (total - *current) * elapsed_time / *current;

		print_progress(*current, total, elapsed_time, time_remaining);

	} while (*current < total);
}

void train_on_images(n_network_t& network, const digit_image_collection_t& training_data_collection, int num_epochs, int batch_size)
{
	int output_idx = network.num_layers - 1;
	int left_idx = network.num_layers - 2;

	//nn_state_t& desired_changes = get_empty_state(network);

	std::cout << "Training network..." << std::endl;

	std::vector<nn_state_t*> desired_changes;
	for (int i = 0; i < batch_size; i++)
	{
		desired_changes.push_back(get_empty_state(network.num_layers, network.layer_sizes));
	}

	batch_handler_t& batch_handler = get_new_batch_handler(training_data_collection, batch_size);
	int epoch_idx = 0;
	std::thread progress_thread(progress_thread, &epoch_idx, num_epochs);

	progress_thread.detach();

	for (; epoch_idx < num_epochs; epoch_idx++)
	{
		digit_image_collection_t training_batch = get_batch(batch_handler);

		//std::cout << "Epoch " << epoch_idx << "/" << num_epochs << std::endl;

		int current_image_idx = 0;
		for each (const digit_image_t & curr in training_batch)
		{
			//std::cout << "image idx " << current_image_idx << std::endl;
			set_input(network, curr);
			feed_forward(network);

			//could do this outside of the loop
			float* unhappiness_for_next_layer = new float[network.layer_sizes[output_idx]];
			for (int i = 0; i < network.layer_sizes[output_idx]; i++)
			{
				//current output node activation
				float activation = network.activations[get_activation_idx(network.state->activation_idx_helper, output_idx, i)];
				//the expected value for the current output node
				float expected = 0.0f;
				if (curr.label == std::to_string(i))
				{
					expected = 1.0f;
				}
				//the bias of the current output node
				float bias = network.state->biases[get_bias_idx(network.state->bias_idx_helper, output_idx, i)];

				//cost derivative
				float input_without_activation_function = logit(activation);

				//cost derivative
				float unhappiness = cost_derivative(activation, expected);
				//activation function (sigmoid) derivative
				float d_sigmoid = sigmoid_derivative(input_without_activation_function);

				//calculate unhappiness
				for (int j = 0; j < network.layer_sizes[left_idx]; j++)
				{
					float unhappiness_weight =
						unhappiness *
						d_sigmoid *
						network.activations[get_activation_idx(network.state->activation_idx_helper, left_idx, j)];

					desired_changes[current_image_idx]->
						weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, output_idx, i, j)]
						+= unhappiness_weight;
				}
				//this is how much the bias should change
				float unhappiness_bias = unhappiness * d_sigmoid;

				desired_changes[current_image_idx]->
					biases[get_bias_idx(network.state->bias_idx_helper, output_idx, i)]
					+= unhappiness_bias;

				unhappiness_for_next_layer[i] = unhappiness * d_sigmoid;
			}
			backprop(network, output_idx - 1, unhappiness_for_next_layer, network.layer_sizes[output_idx], desired_changes, current_image_idx);

			delete[] unhappiness_for_next_layer;

			current_image_idx++;
		}
		nn_state_t* sum_desired_changes = get_empty_state(network.num_layers, network.layer_sizes);
		//calculate average desired change
		for (int i = 0; i < desired_changes.size(); i++)
		{
			for (int j = 1; j < network.num_layers; j++)
			{
				for (int k = 0; k < network.layer_sizes[j]; k++)
				{
					for (int l = 0; l < network.layer_sizes[j - 1]; l++)
					{
						sum_desired_changes->weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, j, k, l)] +=
							desired_changes[i]->weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, j, k, l)];
					}
					sum_desired_changes->biases[get_bias_idx(network.state->bias_idx_helper, j, k)] +=
						desired_changes[i]->biases[get_bias_idx(network.state->bias_idx_helper, j, k)];
				}
			}
		}
		//apply to network (the values get subracted because we want to go in the opposite direction of the gradient)
		for (int j = 1; j < network.num_layers; j++)
		{
			for (int k = 0; k < network.layer_sizes[j]; k++)
			{
				for (int l = 0; l < network.layer_sizes[j - 1]; l++)
				{
					network.state->weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, j, k, l)] -=
						sum_desired_changes->weights[get_weight_idx(network.state->activation_idx_helper, network.layer_sizes, j, k, l)] / desired_changes.size();
				}
				network.state->biases[get_bias_idx(network.state->bias_idx_helper, j, k)] -=
					sum_desired_changes->biases[get_bias_idx(network.state->bias_idx_helper, j, k)] / desired_changes.size();
			}
		}
		for (int i = 0; i < desired_changes.size(); i++)
		{
			clear_state(*desired_changes[i]);
		}
	}
	for (int i = 0; i < desired_changes.size(); i++)
	{
		delete_nn_state(desired_changes[i]);
	}
	//wait for progress thread to finish
	if (progress_thread.joinable())
	{
		progress_thread.join();
	}
}

inline std::string get_file_name(std::string file)
{
	return "../save/" + file + ".state";
}

bool saved_network_exists(std::string file_path)
{
	std::ifstream file(get_file_name(file_path));
	return file.good();
}

void save_network(n_network_t& network, std::string file_path)
{
	std::cout << "Save network " << file_path << std::endl;

	//start output file stream
	std::ofstream out(get_file_name(file_path), std::ios::out | std::ios::binary);

	//write number of layers
	out.write(reinterpret_cast<char*>(&network.num_layers), sizeof(int));

	//write layer sizes
	out.write(reinterpret_cast<char*>(network.layer_sizes), sizeof(int) * network.num_layers);

	//create buffer. this way we dont have to call write that often, which is very costly
	
	int weight_buffer_size = network.state->total_weights;
	float* write_buffer_weight = new float[weight_buffer_size];
	int bias_buffer_size = network.state->total_biases;
	float* write_buffer_bias = new float[bias_buffer_size];

	for (int i = 0; i < weight_buffer_size; i++)
	{
		write_buffer_weight[i] = network.state->weights[i];
	}
	for (int i = 0; i < bias_buffer_size; i++)
	{
		write_buffer_bias[i] = network.state->biases[i];
	}

	//write the buffer to the file
	out.write(reinterpret_cast<const char*>(write_buffer_weight), weight_buffer_size * sizeof(float));
	out.write(reinterpret_cast<const char*>(write_buffer_bias), bias_buffer_size * sizeof(float));

	//delete the buffer
	delete[] write_buffer_weight;
	delete[] write_buffer_bias;

	//close the output stream
	out.close();
}

n_network_t* load_network(std::string file_path)
{
	std::cout << "Loading network " << file_path << std::endl;

	//opoen input file stream
	std::ifstream in(get_file_name(file_path), std::ios::in | std::ios::binary);

	//read how many layers there are
	int num_layers;
	in.read(reinterpret_cast<char*>(&num_layers), sizeof(int));

	//read the sizes of the layers
	int* layer_sizes = new int[num_layers];
	in.read(reinterpret_cast<char*>(layer_sizes), num_layers * sizeof(int));

	//print the layer sizes
	std::cout << "Layer sizes: ";
	for (int i = 0; i < num_layers; i++)
	{
		std::cout << layer_sizes[i] << " ";
	}
	std::cout << std::endl;

	std::vector<int> hidden_layer_sizes;

	for (int i = 1; i < num_layers - 1; i++)
	{
		hidden_layer_sizes.push_back(layer_sizes[i]);
	}

	std::vector<std::string> labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
	if (layer_sizes[num_layers - 1] != labels.size())
	{
		std::cerr << "Error: The number of labels does not match the number of output neurons" << std::endl;
		exit(1);
	}

	//create a new neural network
	//CAUTION this only works for networks with the same hidden layer sizes
	n_network_t* retVal = create_network(layer_sizes[0], hidden_layer_sizes, labels);
	delete[] layer_sizes;

	//create buffer. this way we dont have to call read that often, which is very costly
	int weight_buffer_size = retVal->state->total_weights;
	float* read_buffer_weight = new float[weight_buffer_size];
	int bias_buffer_size = retVal->state->total_biases;
	float* read_buffer_bias = new float[bias_buffer_size];

	//read the data from the file
	in.read(reinterpret_cast<char*>(read_buffer_weight), weight_buffer_size * sizeof(float));
	in.read(reinterpret_cast<char*>(read_buffer_bias), bias_buffer_size * sizeof(float));

	//copy the data from the buffer to the network
	for (int i = 0; i < weight_buffer_size; i++)
	{
		retVal->state->weights[i] = read_buffer_weight[i];
	}
	for (int i = 0; i < bias_buffer_size; i++)
	{
		retVal->state->biases[i] = read_buffer_bias[i];
	}

	//delete the buffer
	delete[] read_buffer_weight;
	delete[] read_buffer_bias;

	//close the input stream
	in.close();

	return retVal;
}

void print_weights(n_network_t& network)
{
	std::cout
		<< "------------------------\n"
		<< "Weights:\n"
		<< "------------------------\n";

	for (int curr_layer = 1; curr_layer < network.num_layers; curr_layer++)
	{
		std::cout << "Layer " << curr_layer << std::endl;
		for (int curr_neuron = 0; curr_neuron < network.layer_sizes[curr_layer]; curr_neuron++)
		{
			for (int left_neuron = 0; left_neuron < network.layer_sizes[curr_layer - 1]; left_neuron++)
			{
				int idx = get_weight_idx(
					network.state->activation_idx_helper,
					network.layer_sizes,
					curr_layer,
					curr_neuron,
					left_neuron);
				std::cout
					<< "layer:" << curr_layer
					<< " curr_n:" << curr_neuron
					<< " left_n:" << left_neuron
					<< " idx: " << idx
					<< " value:" << network.state->weights[idx] << " \n";

			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

void print_biases(n_network_t& network)
{
	std::cout
		<< "------------------------\n"
		<< "Biases:\n"
		<< "------------------------\n";

	for (int curr_layer = 1; curr_layer < network.num_layers; curr_layer++)
	{
		std::cout << "Layer " << curr_layer << std::endl;
		for (int curr_neuron = 0; curr_neuron < network.layer_sizes[curr_layer]; curr_neuron++)
		{
			int idx = get_bias_idx(network.state->bias_idx_helper, curr_layer, curr_neuron);
			std::cout
				<< "layer: " << curr_layer << " neuron: " << curr_neuron << "  "
				<< "idx " << idx << ": "
				<< network.state->biases[idx] << " \n";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}