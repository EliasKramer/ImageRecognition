#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>

constexpr int IMAGE_SIZE_X = 28;
constexpr int IMAGE_SIZE_Y = 28;

typedef struct {
    float matrix[IMAGE_SIZE_Y][IMAGE_SIZE_X];
    std::string label;
} digit_image;

typedef std::vector<digit_image> digit_image_collection;

void print_digit_image(digit_image& data);
digit_image_collection load_mnist_data(std::string data_file_path, std::string label_file_path);