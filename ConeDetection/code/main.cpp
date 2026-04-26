/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: The due date
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

#include "ConeFilter.h"

using namespace std;
namespace fs = filesystem;
using namespace fs;

const path out_path = "filtered_images";

int main(int argc, char** argv)
{   
    create_directories(out_path);

    // -------------------- FILE PARSING --------------------

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

    // -------------------- IMAGE / ANNOTATION VECTOR --------------------
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

    // -------------------- IMAGE FILTERING --------------------
    path filtered_img_path = out_path / dataset_path.filename();
    create_directory(filtered_img_path);

    for (const path& image_path : images)
    {
        int width = 0;
        int height = 0;
        int channels = 0;

        unsigned char* pixels = stbi_load(image_path.string().c_str(),
                                      &width, &height, &channels, 0);

        const int min_height_bound = height * 0.5;
        const int max_height_bound = height * 0.8;

        if (!pixels) 
        {
            cerr << "Failed to load image: " << image_path << "\n";
            return 1;
        }

        vector<unsigned char> output(width * height * 3, 0);

        for (int j = min_height_bound; j < max_height_bound; j++)
        {
            for (int i = 0; i < width; i++)
            {
                int pixel_index = (j * width) + i;

                unsigned char r = pixels[pixel_index * 3 + 0];
                unsigned char g = pixels[pixel_index * 3 + 1];
                unsigned char b = pixels[pixel_index * 3 + 2];

                // if (is_orange_cone(r, g, b))
                // {
                //     output[pixel_index * 3 + 0] = 255;
                //     output[pixel_index * 3 + 1] = 120;
                //     output[pixel_index * 3 + 2] = 0;
                // }
                // else if (is_yellow_cone(r, g, b))
                // {
                //     output[pixel_index * 3 + 0] = 255;
                //     output[pixel_index * 3 + 1] = 255;
                //     output[pixel_index * 3 + 2] = b;
                // }
                // else if (is_blue_cone(r, g, b))
                // {
                //     output[pixel_index * 3 + 0] = r;
                //     output[pixel_index * 3 + 1] = g;
                //     output[pixel_index * 3 + 2] = 255;
                // }

                ConeColor color = classify_pixel(r, g, b);

                if (color == ConeColor::Yellow)
                {
                    output[pixel_index * 3 + 0] = 255;
                    output[pixel_index * 3 + 1] = 255;
                    output[pixel_index + 2] = 0;
                }
                else if (color == ConeColor::Orange)
                {
                    output[pixel_index * 3 + 0] = 255;
                    output[pixel_index * 3+ 1] = 120;
                    output[pixel_index * 3 + 2] = 0;
                }
                else if (color == ConeColor::Blue)
                {
                    output[pixel_index * 3 + 0] = 0;
                    output[pixel_index * 3 + 1] = 0;
                    output[pixel_index * 3 + 2] = 255;
                }
                else
                {
                    output[pixel_index * 3 + 0] = r / 4;
                    output[pixel_index * 3 + 1] = g / 4;
                    output[pixel_index * 3 + 2] = b / 4;
                }
            }
        }

        path output_file = filtered_img_path / image_path.filename();
        output_file.replace_extension(".png");

        stbi_write_png(
            output_file.string().c_str(),
            width,
            height,
            3,
            output.data(),
            width * 3
        );

        cout << "Wrote: " << output_file << endl;

        stbi_image_free(pixels);
    }

    return 0;
}
