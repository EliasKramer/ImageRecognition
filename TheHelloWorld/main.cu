
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "data.h"
#include "constants.h"
#include "network.h"

int main()
{
	std::cout << "Hello World" << std::endl;

	//input_picture* x = create_struct_from_file(TRAINING_DATA_PATH + "digit_0\\4558.png");
	//print_picture(*x);
	neural_network* nn = create_neural_network(16, 16);
	
	
	return 0;
}