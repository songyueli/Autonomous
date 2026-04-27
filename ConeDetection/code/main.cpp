#include <iostream>
#include <algorithm>
#include <string>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ConeFilter.h"

using namespace std;
namespace fs = filesystem;

const fs::path out_path = "filtered_images";

// DBSCAN parameters
const float eps = 15.0f;        // Maximum distance between points in a cluster
const int min_pts = 10;         // Minimum points to form a cluster

// Draw rectangle on image
void draw_rectangle(
    vector<unsigned char>& image,
    int width,
    int height,
    const BoundingBox& box,
    unsigned char r,
    unsigned char g,
    unsigned char b,
    int thickness = 2
) {
    // Draw top and bottom edges
    for (int x = box.x_min; x <= box.x_max; x++) {
        if (x < 0 || x >= width) continue;
        
        for (int t = 0; t < thickness; t++) {
            // Top edge
            int y_top = box.y_min + t;
            if (y_top >= 0 && y_top < height) {
                int idx = (y_top * width + x) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
            
            // Bottom edge
            int y_bot = box.y_max - t;
            if (y_bot >= 0 && y_bot < height) {
                int idx = (y_bot * width + x) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
        }
    }
    
    // Draw left and right edges
    for (int y = box.y_min; y <= box.y_max; y++) {
        if (y < 0 || y >= height) continue;
        
        for (int t = 0; t < thickness; t++) {
            // Left edge
            int x_left = box.x_min + t;
            if (x_left >= 0 && x_left < width) {
                int idx = (y * width + x_left) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
            
            // Right edge
            int x_right = box.x_max - t;
            if (x_right >= 0 && x_right < width) {
                int idx = (y * width + x_right) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
        }
    }
}

int main(int argc, char** argv) {
    create_directories(out_path);

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <dataset_path>\n";
        return 1;
    }

    fs::path dataset_path = argv[1];

    if (!fs::exists(dataset_path)) {
        cerr << "ERROR: File path does not exist\n";
        return 1;
    }

    cout << "Using dataset: " << dataset_path << endl;

    fs::path ann_path = dataset_path / "ann";
    fs::path img_path = dataset_path / "img";

    if (!exists(ann_path) || !is_directory(ann_path)) {
        throw runtime_error("ann folder missing");
    }

    if (!exists(img_path) || !is_directory(img_path)) {
        throw runtime_error("img folder missing");
    }

    vector<fs::path> images;

    for (const auto& entry : fs::directory_iterator(img_path)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png")
                images.push_back(entry.path());
        }
    }

    sort(images.begin(), images.end());

    fs::path filtered_img_path = out_path / dataset_path.filename();
    create_directory(filtered_img_path);

    for (const fs::path& image_path : images) {
        int width = 0, height = 0, channels = 0;

        unsigned char* pixels = stbi_load(
            image_path.string().c_str(),
            &width, &height, &channels, 0
        );

        if (!pixels) {
            cerr << "Failed to load image: " << image_path << "\n";
            continue;
        }

        const int min_height_bound = height * 0.3;
        const int max_height_bound = height * 0.8;

        // ANALYZE SCENE BRIGHTNESS
        SceneStats scene = analyze_scene(pixels, width, height, 
                                        min_height_bound, max_height_bound);
        
        cout << "Image: " << image_path.filename() 
            << " (avg brightness: " << scene.avg_brightness 
            << ", dark: " << (scene.is_dark_scene ? "YES" : "NO") << ")" << endl;

        // Collect colored points with scene-aware classification
        vector<Point> yellow_points, orange_points, blue_points;

        // Create output image (dimmed background)
        vector<unsigned char> output(width * height * 3, 0);

        for (int j = min_height_bound; j < max_height_bound; j++) {
            for (int i = 0; i < width; i++) {
                int pixel_index = (j * width) + i;
                int channel_index = pixel_index * 3;

                unsigned char r = pixels[channel_index + 0];
                unsigned char g = pixels[channel_index + 1];
                unsigned char b = pixels[channel_index + 2];

                // Pass scene info to classifier
                ConeColor color = classify_pixel(r, g, b, scene.is_dark_scene);

                if (color == ConeColor::Yellow) {
                    yellow_points.push_back({i, j, ConeColor::Yellow});
                    output[channel_index + 0] = 255;
                    output[channel_index + 1] = 255;
                    output[channel_index + 2] = 0;
                } else if (color == ConeColor::Orange) {
                    orange_points.push_back({i, j, ConeColor::Orange});
                } else if (color == ConeColor::Blue) {
                    blue_points.push_back({i, j, ConeColor::Blue});
                    output[channel_index + 0] = 0;
                    output[channel_index + 1] = 0;
                    output[channel_index + 2] = 255;
                }
            }
        }

        // Adaptive DBSCAN parameters based on scene
        float eps = scene.is_dark_scene ? 12.0f : 10.0f;  // Larger eps in dark
        int min_pts = scene.is_dark_scene ? 10 : 15;       // Fewer points needed in dark

        // Rest of clustering code stays the same...
        vector<BoundingBox> yellow_boxes = cluster_cones(yellow_points, eps, min_pts);
        vector<BoundingBox> orange_boxes = cluster_cones(orange_points, eps, min_pts);
        vector<BoundingBox> blue_boxes = cluster_cones(blue_points, eps, min_pts);
        

        yellow_boxes = merge_overlapping_boxes(yellow_boxes);
        orange_boxes = merge_overlapping_boxes(orange_boxes);
        blue_boxes = merge_overlapping_boxes(blue_boxes);

        cout << "Image: " << image_path.filename() << endl;
        cout << "  Yellow cones: " << yellow_boxes.size() << endl;
        cout << "  Orange cones: " << orange_boxes.size() << endl;
        cout << "  Blue cones: " << blue_boxes.size() << endl;

        // for (int j = 0; j < height; j++) {
        //     for (int i = 0; i < width; i++) {
        //         int pixel_index = (j * width) + i;
        //         int channel_index = pixel_index * 3;

        //         output[channel_index + 0] = pixels[channel_index + 0] / 4;
        //         output[channel_index + 1] = pixels[channel_index + 1] / 4;
        //         output[channel_index + 2] = pixels[channel_index + 2] / 4;
        //     }
        // }

        // Draw bounding boxes
        for (const auto& box : yellow_boxes) {
            draw_rectangle(output, width, height, box, 255, 255, 0, 3);
        }
        for (const auto& box : orange_boxes) {
            draw_rectangle(output, width, height, box, 255, 120, 0, 3);
        }
        for (const auto& box : blue_boxes) {
            draw_rectangle(output, width, height, box, 0, 100, 255, 3);
        }

        fs::path output_file = filtered_img_path / image_path.filename();
        output_file.replace_extension(".png");

        stbi_write_png(
            output_file.string().c_str(),
            width, height, 3,
            output.data(),
            width * 3
        );

        cout << "  Wrote: " << output_file << endl;

        stbi_image_free(pixels);
    }

    return 0;
}