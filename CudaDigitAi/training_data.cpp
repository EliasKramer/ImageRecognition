#include "training_data.h"
int swap_endian(int value) {
    int result = 0;
    result |= (value & 0xFF) << 24;
    result |= ((value >> 8) & 0xFF) << 16;
    result |= ((value >> 16) & 0xFF) << 8;
    result |= ((value >> 24) & 0xFF);
    return result;
}
//determins wether the system is little endian or big endian
bool is_little_endian()
{
    int num = 1;
	return (*(char*)&num == 1);
}
void print_training_data(digit_image& data)
{
    //print the two dimensional float array of the image in the according colors
    std::cout << std::endl << "-------------------------------------" << std::endl;
    std::cout << "Label: " << data.label << std::endl;
    for (int y = 0; y < data.rows; y++)
    {
        for (int x = 0; x < data.cols; x++)
        {
            if (data.matrix[y][x] == 0)
            {
				std::cout << "  ";
			}
            else if (data.matrix[y][x] < 0.5)
            {
				std::cout << ". ";
			}
            else
            {
				std::cout << "# ";
			}
		}
		std::cout << std::endl;
	}
    std::cout << std::endl << "-------------------------------------" << std::endl;

}

digit_image_collection load_mnist_data(std::string data_file_path, std::string label_file_path) {
    digit_image_collection mnist_data;
    // Check if data file exists
    std::ifstream data_file1(data_file_path, std::ios::binary);
    
    //get actual path used
    //std::filesystem::path path = std::filesystem::current_path();

    if (!data_file1) {
        std::cerr << "Data file does not exist" << std::endl;
        exit(1);
    }

    // Check if label file exists
    std::ifstream label_file1(label_file_path, std::ios::binary);
    if (!label_file1) {
        std::cerr << "Label file does not exist" << std::endl;
        exit(1);
    }

    // Open the data file and read the magic number and number of images
    std::ifstream data_file(data_file_path, std::ios::binary);
    int magic_number, num_images, rows, cols;
    data_file.read((char*)&magic_number, sizeof(magic_number));
    data_file.read((char*)&num_images, sizeof(num_images));
    data_file.read((char*)&rows, sizeof(rows));
    data_file.read((char*)&cols, sizeof(cols));

    if (is_little_endian())
    {
        magic_number = swap_endian(magic_number);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);
    }

    // Open the label file and read the magic number and number of labels
    std::ifstream label_file(label_file_path, std::ios::binary);
    int label_magic_number, num_labels;
    label_file.read((char*)&label_magic_number, sizeof(label_magic_number));
    label_file.read((char*)&num_labels, sizeof(num_labels));

    if (is_little_endian())
    {
        label_magic_number = swap_endian(label_magic_number);
        num_labels = swap_endian(num_labels);
    }

    // Check that the magic numbers and number of items match
    if (magic_number != 2051 || label_magic_number != 2049 || num_images != num_labels) {
        std::cout << "Error: Invalid MNIST data files" << std::endl;
        return mnist_data;
    }

    // Read each image and label and store them in a training_data struct
    for (int i = 0; i < num_images; i++) {
        float** image = new float* [rows];
        for (int j = 0; j < rows; j++) {
            image[j] = new float[cols];
            for (int k = 0; k < cols; k++) {
                unsigned char pixel;
                data_file.read((char*)&pixel, sizeof(pixel));
                image[j][k] = (float)pixel / 255.0;
            }
        }
        unsigned char label;
        label_file.read((char*)&label, sizeof(label));
        digit_image data;
        data.matrix = image;
        data.rows = rows;
        data.cols = cols;
        data.label = std::to_string(label);
        mnist_data.push_back(data);
    }


    // Close the files and return the data
    data_file.close();
    label_file.close();
    return mnist_data;
}