#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "training_data.hpp"
#include "neural_network.hpp"

#pragma once

void cuda_train_on_images(
	n_network_t* network,
	digit_image_collection_t& images,
	int epochs,
	int batch_size);