#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "cuda_single_nn.hpp"

__device__ int get_activation_idx_cuda(int* activation_idx_helper, int curr_layer, int curr_neuron)
{
	return activation_idx_helper[curr_layer] + curr_neuron;
}
__device__ int get_bias_idx_cuda(int* bias_idx_helper, int curr_layer, int curr_neuron)
{
	return bias_idx_helper[curr_layer - 1] + curr_neuron;
}
__device__ int get_weight_idx_cuda(int* activation_idx_helper, int* layer_sizes, int curr_layer, int curr_neuron, int left_neuron)
{
	return get_activation_idx_cuda(activation_idx_helper, curr_layer - 1, curr_neuron) * layer_sizes[curr_layer - 1] + left_neuron;
}
__device__ float sigmoid_cuda(float x)
{
	return 1 / (1 + exp(-x));
}

__global__ void feed_forward(
	int layer, 
	int* layer_sizes,
	float* weights, 
	float* biases, 
	float* activations, 
	int* activation_idx_helper, 
	int* bias_idx_helper
	)
{
    int i = threadIdx.x;

	int activation_idx = get_activation_idx_cuda(activation_idx_helper, layer, i);
	int bias_idx = get_bias_idx_cuda(bias_idx_helper, layer, i);
	for (int j = 0; j < layer_sizes[layer - 1]; j++)
	{
		int weight_idx = get_weight_idx_cuda(activation_idx_helper, layer_sizes, layer, i, j);
		activations[activation_idx] += weights[weight_idx] * activations[get_activation_idx_cuda(activation_idx_helper, layer - 1, j)];
	}
	activations[activation_idx] += biases[bias_idx];
	activations[activation_idx] = sigmoid_cuda(activations[activation_idx]);
}

template<typename T>
static T* copy_arr_to_gpu(T* host_data, int size)
{
	T* device_data;
	cudaError_t cudaStatus = cudaMalloc((void**)&device_data, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		std::cout << "allocating memory on the gpu failed!";
		exit(1);
	}
	cudaStatus = cudaMemcpy(device_data, host_data, sizeof(T) * size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "copying memory to the gpu failed!";
		exit(1);
	}
	return device_data;
}

void cuda_train_on_images(n_network_t* network, digit_image_collection_t& images, int epochs, int batch_size)
{
	set_input(*network, images.at(0));

	float* device_activations = copy_arr_to_gpu<float>(network->activations, network->state->total_nodes);
	float* device_weights = copy_arr_to_gpu<float>(network->state->weights, network->state->total_weights);
	float* device_biases = copy_arr_to_gpu<float>(network->state->biases, network->state->total_biases);
	
	int* device_layer_sizes = copy_arr_to_gpu<int>(network->layer_sizes, network->num_layers);
	int* device_activation_idx_helper = copy_arr_to_gpu<int>(network->state->activation_idx_helper, network->num_layers);
	int* device_bias_idx_helper = copy_arr_to_gpu<int>(network->state->bias_idx_helper, network->num_layers);

	//start timer
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (int i = 1; i < network->num_layers; i++)
	{
		feed_forward << <1, network->layer_sizes[i] >> > (
			i, //current layer 
			device_layer_sizes,
			device_weights,
			device_biases,
			device_activations,
			device_activation_idx_helper,
			device_bias_idx_helper);
		//wait for process to finish
	}
	cudaDeviceSynchronize();
	//copy output activations to host
	float* copy_of_device_activations = new float[network->state->total_nodes];
	cudaMemcpy(copy_of_device_activations, device_activations, sizeof(float) * network->state->total_nodes, cudaMemcpyDeviceToHost);

	//stop timer
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	//print time
	std::cout << "time taken gpu : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	
	//start timer
	start = std::chrono::high_resolution_clock::now();
	feed_forward(*network);
	//stop timer
	end = std::chrono::high_resolution_clock::now();

	//print time
	std::cout << "time taken cpu : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//print if they are the same
	for (int i = 1; i < network->num_layers; i++)
	{
		std::cout << "layer : " << i << std::endl;
		for (int j = 0; j < network->layer_sizes[i]; j++)
		{
			//print both activations
			std::cout 
				<< copy_of_device_activations[get_activation_idx(network->state->activation_idx_helper, i, j)] << " " 
				<< network->activations[get_activation_idx(network->state->activation_idx_helper, i, j)] << std::endl;
		}
		std::cout << std::endl;
	}

	//free mem
	cudaFree(device_activations);
	cudaFree(device_weights);
	cudaFree(device_biases);
	cudaFree(device_layer_sizes);
	cudaFree(device_activation_idx_helper);
	cudaFree(device_bias_idx_helper);
	delete[] copy_of_device_activations;
}
