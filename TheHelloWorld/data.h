#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <lodepng.h>
#include "constants.h"

typedef struct {
	float pixel_arr[INPUT_PIC_DIMENSION_X][INPUT_PIC_DIMENSION_Y];
} input_picture;

input_picture* create_input_picture();

input_picture* create_struct_from_file(const std::string& filename);

void print_picture(input_picture& pic);

void print_string_in_gray_value(const std::string& str, float intensity);