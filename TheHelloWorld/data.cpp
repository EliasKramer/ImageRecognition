#include "data.h"


input_picture* create_input_picture()
{
	return (input_picture*)malloc(sizeof(input_picture));
}

input_picture* create_struct_from_file(const std::string& filename)
{
	std::vector<unsigned char> image;
	unsigned width, height;
	unsigned error = lodepng::decode(image, width, height, filename);
	if (error) {
		throw std::runtime_error("Error " + std::to_string(error) + ": " + lodepng_error_text(error) + " (file path: " + filename + ")");
		return nullptr;
	}
	if (width != INPUT_PIC_DIMENSION_X || height != INPUT_PIC_DIMENSION_Y)
	{
		throw std::runtime_error("Error: Image dimensions do not match expected dimensions (file path: " + filename + ")");
		return nullptr;
	}

	// The image is now stored in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
	input_picture* ret_val = create_input_picture();
	for (size_t i = 0; i < image.size(); i += 4) {
		// Get the grayscale value by averaging the R, G, and B values
		float value = (float)((float)image[i] + (float)image[i + 1] + (float)image[i + 2]) / (float)3;
		ret_val->pixel_arr[i / 4 / INPUT_PIC_DIMENSION_X][i / 4 % INPUT_PIC_DIMENSION_X] = value;
	}

	return ret_val;
}

void print_picture(input_picture& pic)
{
	for (int i = 0; i < INPUT_PIC_DIMENSION_X; i++)
	{
		for (int j = 0; j < INPUT_PIC_DIMENSION_Y; j++)
		{
			std::cout << pic.pixel_arr[i][j] << " ";
		}
		std::cout << std::endl;
	}
}
