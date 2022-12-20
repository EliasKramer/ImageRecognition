#pragma once

#include "node.h"
#include <iostream>
#include <string>
#include <vector>
#include <lodepng.h>

const unsigned int INPUT_PIC_DIMENSION_X = 32;
const unsigned int INPUT_PIC_DIMENSION_Y = 32;

typedef struct {
	float pixel_arr[INPUT_PIC_DIMENSION_X][INPUT_PIC_DIMENSION_Y];
} input_picture;

input_picture* create_input_picture();

input_picture* create_struct_from_file(const std::string& filename);

void print_picture(input_picture& pic);