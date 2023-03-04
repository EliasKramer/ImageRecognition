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
void print_digit_image(digit_image_t& data)
{
    //print the two dimensional float array of the image in the according colors
    std::cout << std::endl << "-------------------------------------" << std::endl;
    std::cout << "Label: " << data.label << std::endl;
    for (int y = 0; y < IMAGE_SIZE_Y; y++)
    {
        for (int x = 0; x < IMAGE_SIZE_X; x++)
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

digit_image_collection_t load_mnist_data(std::string data_file_path, std::string label_file_path) {
    digit_image_collection_t mnist_data;
    
    //get the path where the program is executed
    std::filesystem::path path = std::filesystem::current_path();
    //go to folder where the data directory is located
    path = path.parent_path();
    
    //replace all backslashes with forward slashes
    std::string base_path = path.string();
    std::replace(base_path.begin(), base_path.end(), '\\', '/');

    std::string full_data_path = base_path + data_file_path;
    std::string full_label_path = base_path + label_file_path;
    
    std::cout << "reading images from " << full_data_path << std::endl;
    std::cout << "reading labels from " << full_label_path << std::endl;

    //check if files exists
    if (!std::filesystem::exists(std::filesystem::path(full_data_path)) ||
        !std::filesystem::exists(std::filesystem::path(full_label_path)))
    {
        std::cerr << "A file does not exist" << std::endl;
        exit(1);
    }

    //Open the data file and read the magic number and number of images
    //The magic number is there to check if the file is read correctly
    std::ifstream data_file(full_data_path, std::ios::binary);
    int magic_number, num_images, rows, cols;
    //can be improved with structs
    data_file.read((char*)&magic_number, sizeof(magic_number));
    data_file.read((char*)&num_images, sizeof(num_images));
    data_file.read((char*)&rows, sizeof(rows));
    data_file.read((char*)&cols, sizeof(cols));

    //the magic number is stored in big endian, 
    //so we need to swap the bytes, 
    //if we are on a little endian system
    if (is_little_endian())
    {
        magic_number = swap_endian(magic_number);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);
    }

    // Open the label file and read the magic number and number of labels
    std::ifstream label_file(full_label_path, std::ios::binary);
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
        std::cerr << "Error: Invalid MNIST data files" << std::endl;
        exit(1);
    }

    //read all pixel values and labels at once, 
    //because reading from a file is a very costly operation
    int image_buffer_size = num_images * rows * cols;
    char* image_buffer = new char[image_buffer_size];
    data_file.read(image_buffer, image_buffer_size);

    char* label_buffer = new char[num_labels];
    label_file.read(label_buffer, num_labels);

    digit_image_t current_image;

    for (int i = 0; i < num_images; i++) {
        
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {

                int pixel_idx = i * rows * cols + j * cols + k;
                //why is this "reading invalid data from image_buffer" ?
                unsigned char pixel = image_buffer[pixel_idx];

                current_image.matrix[j][k] = (float)pixel / 255.0;
            }
        }

        unsigned char label = label_buffer[i];
        current_image.label = std::to_string(label);

        //push back creates a copy of the object
        mnist_data.push_back(current_image);
    }

    delete[] image_buffer;
    delete[] label_buffer;

    data_file.close();
    label_file.close();

    return mnist_data;
}