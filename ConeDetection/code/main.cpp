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
#include <vector>
#include <stdexcept>
#include <cstdint>

// #include <opencv2/opencv.hpp> // Optional method for opening image

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
namespace fs = filesystem;
using namespace fs;

int main(int argc, char** argv)
{
    // Collect args
    if (argc != 2) 
    {
        cerr << "Usage: " << argv[0] << " <dataset_path>\n";
        return 1;
    }

    path dataset_path = argv[1];

    if (!filesystem::exists(dataset_path))
    {
        cerr << "ERROR: File path does not exist\n";
    }

    cout << "Using dataset: " << dataset_path << endl;

    path ann_path = dataset_path / "ann";
    path img_path = dataset_path / "img";

    // annotation and images directory checks for fsoco
    if (!exists(ann_path) || !is_directory(ann_path)) 
    {
        throw runtime_error("ann folder missing");
    }

    if (!exists(img_path) || !is_directory(img_path)) 
    {
        throw runtime_error("img folder missing");
    }

    // Generate a vector of paths for images
    vector<path> images;

    for (const auto& entry : directory_iterator(img_path))
    {
        if (entry.is_regular_file())
        {
            auto ext = entry.path().extension().string();

            if (ext == ".jpg" || ext == ".png")
                images.push_back(entry.path());
        }
    }

    sort(images.begin(), images.end());
    path first_image = images.front();

    if (first_image.empty()) 
    {
        throw runtime_error("No JPG images found");
    }
    else 
    { 
        // Debug Print
        cout << "Selected first image: " << first_image.string() << endl;
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char* pixels = stbi_load(first_image.string().c_str(),
                                      &width, &height, &channels, 0);

    if (!pixels) 
    {
        cerr << "Failed to load image: " << first_image << "\n";
        return 1;
    }

    cout << "Loaded image successfully\n";
    cout << "Width: " << width
              << ", Height: " << height
              << ", Channels: " << channels << "\n";

    path out_file = "debug_first_image.png";

    if (!stbi_write_png(out_file.string().c_str(),
                        width,
                        height,
                        channels,
                        pixels,
                        width * channels))
    {
        cerr << "Failed to save output image\n";
        stbi_image_free(pixels);
        return 1;
    }

    cout << "Saved debug image to: " << out_file << "\n";

    stbi_image_free(pixels);

    return 0;
}
