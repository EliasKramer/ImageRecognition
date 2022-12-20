
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "data.h"

int main()
{
	std::cout << "Hello World" << std::endl;
    
	input_picture* x = create_struct_from_file("..\\Data\\Train\\digit_0\\4558.png");
	print_picture(*x);
	return 0;
}