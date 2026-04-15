/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: Latest
 * Class: ECE 4122
 * 
 * Description: This is the main program
*/

#include <iostream>
#include <algorithm>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp> // Optional method for opening image

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;
namespace fs = filesystem;
using namespace fs;

int main(int argc, char** argv)
{
    // Collect args
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <dataset_path>\n";
        return 1;
    }

    path dataset_path = argv[1];

    if (!filesystem::exists(dataset_path))
    {
        cerr << "ERROR: File path does not exist\n";
    }

    cout << "Hello chuds" << endl; 
    cout << "Using dataset: " << dataset_path << endl;

    path ann_path = dataset_path / "ann";
    path img_path = dataset_path / "img";

    // annotation and images directory checks for fsoco
    if (!exists(ann_path) || !is_directory(ann_path)) {
        throw runtime_error("ann folder missing");
    }

    if (!exists(img_path) || !is_directory(img_path)) {
        throw runtime_error("img folder missing");
    }

    // Check first image
    path first_image;

    for (const auto& entry : directory_iterator(img_path)) {
        if (entry.is_regular_file()) {
            if (entry.path().extension() == ".jpg") {
                first_image = entry.path();
                break;
            }
        }
    }

    if (first_image.empty()) {
        throw std::runtime_error("No JPG images found");
    }

    // Load Image into stb_image
    cv::Mat img = cv::imread(first_image.string());
    cv::imshow("Image", img);
    cv::waitKey(0);

    // Print Information
    std::cout << "Image loaded successfully\n";
    // std::cout << "Width: " << width
    //           << ", Height: " << height
    //           << ", Channels: " << channels << "\n";


    // Exit
    stbi_image_free(data);

    return 0;
}
