#include <iostream>
#include <algorithm>
#include <string>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <omp.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ConeFilter.h"
#include "Draw.h"

using namespace std;
using namespace std::chrono;
namespace fs = filesystem;

const fs::path out_path = "filtered_images";
const int num_images_parse = 100;

// Process a single image (used in parallel loop)
void process_image(const fs::path& image_path, const fs::path& filtered_img_path) 
{
    int width = 0, height = 0, channels = 0;

    unsigned char* pixels = stbi_load(
        image_path.string().c_str(),
        &width, &height, &channels, 0
    );

    if (!pixels) {
        #pragma omp critical
        {
            cerr << "Failed to load image: " << image_path << "\n";
        }
        return;
    }

    const int min_height_bound = height * 0.3;
    const int max_height_bound = height * 0.8;

    // ANALYZE SCENE BRIGHTNESS (parallelized internally)
    SceneStats scene = analyze_scene(pixels, width, height, 
                                    min_height_bound, max_height_bound);
    
    #pragma omp critical
    {
        cout << "Image: " << image_path.filename() 
            << " (avg brightness: " << scene.avg_brightness 
            << ", dark: " << (scene.is_dark_scene ? "YES" : "NO") << ")" << endl;
    }

    // Collect colored points with scene-aware classification
    vector<Point> yellow_points, orange_points, blue_points;

    // Create output image (dimmed background)
    vector<unsigned char> output(width * height * 3, 0);

    // Parallelize pixel classification with OpenMP
    #pragma omp parallel
    {
        // Thread-local storage for points
        vector<Point> local_yellow, local_orange, local_blue;
        
        #pragma omp for nowait schedule(dynamic, 100)
        for (int j = min_height_bound; j < max_height_bound; j++) 
        {
            for (int i = 0; i < width; i++) 
            {
                int pixel_index = (j * width) + i;
                int channel_index = pixel_index * 3;

                unsigned char r = pixels[channel_index + 0];
                unsigned char g = pixels[channel_index + 1];
                unsigned char b = pixels[channel_index + 2];

                ConeColor color = classify_pixel(r, g, b, scene.is_dark_scene);

                if (color == ConeColor::Yellow) 
                {
                    local_yellow.push_back({i, j, ConeColor::Yellow});
                    output[channel_index + 0] = 255;
                    output[channel_index + 1] = 255;
                    output[channel_index + 2] = 0;
                } 
                else if (color == ConeColor::Orange) 
                {
                    local_orange.push_back({i, j, ConeColor::Orange});
                    output[channel_index + 0] = 255;
                    output[channel_index + 1] = 120;
                    output[channel_index + 2] = 0;
                } 
                else if (color == ConeColor::Blue) 
                {
                    local_blue.push_back({i, j, ConeColor::Blue});
                    output[channel_index + 0] = 0;
                    output[channel_index + 1] = 0;
                    output[channel_index + 2] = 255;
                }
            }
        }
        
        // Merge thread-local results into global vectors
        #pragma omp critical
        {
            yellow_points.insert(yellow_points.end(), local_yellow.begin(), local_yellow.end());
            orange_points.insert(orange_points.end(), local_orange.begin(), local_orange.end());
            blue_points.insert(blue_points.end(), local_blue.begin(), local_blue.end());
        }
    }

    // Adaptive DBSCAN parameters based on scene
    float eps = scene.is_dark_scene ? 12.0f : 5.0f;
    int min_pts = scene.is_dark_scene ? 10 : 40;

    // Clustering (could also be parallelized but more complex)
    vector<BoundingBox> yellow_boxes = cluster_cones(yellow_points, eps, min_pts);
    vector<BoundingBox> orange_boxes = cluster_cones(orange_points, eps, min_pts);
    vector<BoundingBox> blue_boxes = cluster_cones(blue_points, eps, min_pts);
    
    yellow_boxes = merge_overlapping_boxes(yellow_boxes);
    orange_boxes = merge_overlapping_boxes(orange_boxes);
    blue_boxes = merge_overlapping_boxes(blue_boxes);

    #pragma omp critical
    {
        cout << "Image: " << image_path.filename() << endl;
        cout << "  Yellow cones: " << yellow_boxes.size() << endl;
        cout << "  Orange cones: " << orange_boxes.size() << endl;
        cout << "  Blue cones: " << blue_boxes.size() << endl;
    }

    // Draw bounding boxes
    for (const auto& box : yellow_boxes) 
    {
        draw_rectangle(output, width, height, box, 255, 255, 0, 3);
    }
    for (const auto& box : orange_boxes) 
    {
        draw_rectangle(output, width, height, box, 255, 120, 0, 3);
    }
    for (const auto& box : blue_boxes) 
    {
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

    #pragma omp critical
    {
        cout << "  Wrote: " << output_file << endl;
    }

    stbi_image_free(pixels);
}

int main(int argc, char** argv) 
{
    create_directories(out_path);

    if (argc != 2) 
    {
        cerr << "Usage: " << argv[0] << " <dataset_path>\n";
        return 1;
    }

    fs::path dataset_path = argv[1];

    if (!fs::exists(dataset_path)) 
    {
        cerr << "ERROR: File path does not exist\n";
        return 1;
    }

    cout << "Using dataset: " << dataset_path << endl;

    fs::path ann_path = dataset_path / "ann";
    fs::path img_path = dataset_path / "img";

    if (!exists(ann_path) || !is_directory(ann_path)) 
    {
        throw runtime_error("ann folder missing");
    }

    if (!exists(img_path) || !is_directory(img_path)) 
    {
        throw runtime_error("img folder missing");
    }

    vector<fs::path> images;

    for (const auto& entry : fs::directory_iterator(img_path)) 
    {
        if (entry.is_regular_file()) 
        {
            auto ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png")
            {
                images.push_back(entry.path());
            }
        }
    }

    sort(images.begin(), images.end());

    // Limit to num_images_parse
    if (images.size() > num_images_parse) 
    {
        images.resize(num_images_parse);
    }

    fs::path filtered_img_path = out_path / dataset_path.filename();
    create_directory(filtered_img_path);

    // Set number of threads for work INSIDE each image
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    cout << "Processing images one at a time\n";
    cout << "Using up to " << num_threads 
        << " threads inside each image\n";

    for (size_t i = 0; i < images.size(); i++) 
    {
        auto process_start = high_resolution_clock::now();
        
        cout << "\nProcessing image " << (i + 1)
            << " / " << images.size()
            << ": " << images[i].filename() << endl;

        process_image(images[i], filtered_img_path);

        auto process_end = high_resolution_clock::now();
        auto process_time = duration_cast<microseconds>(process_end - process_start).count();

        cout << "Processing Time: " << process_time / 1000.0 << endl;
    }

    return 0;
}