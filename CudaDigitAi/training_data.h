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
} training_data;

void print_training_data(training_data& data);
std::vector<training_data> load_mnist_data(std::string data_file_path, std::string label_file_path);