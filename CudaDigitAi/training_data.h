#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>

typedef struct {
    float** matrix;
    int rows;
    int cols;
    std::string label;
} digit_image;

typedef std::vector<digit_image> digit_image_collection;

void print_training_data(digit_image& data);
digit_image_collection load_mnist_data(std::string data_file_path, std::string label_file_path);