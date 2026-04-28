/**
 * Real-time sequential image processing with intra-image parallelization
 * Mimics actual autonomous vehicle operation: images arrive one at a time
 * Parallelization happens WITHIN each image, not across multiple images
 */

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
#include "ConeFilterCUDA.cuh"

using namespace std;
using namespace std::chrono;
namespace fs = filesystem;

const fs::path out_path = "filtered_images";
const int num_images_parse = 100;

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
    // Parallelize rectangle drawing for large boxes
    #pragma omp parallel for if(box.x_max - box.x_min > 100)
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
    
    #pragma omp parallel for if(box.y_max - box.y_min > 100)
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

    // Set number of threads for intra-image parallelization
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads for intra-image parallelization\n";
    cout << "Processing images SEQUENTIALLY (mimicking real-time autonomous vehicle)\n";
    cout << "=========================================\n";

    int img_count = 0;
    
    // SEQUENTIAL image processing (mimics real-time stream)
    for (const fs::path& image_path : images) {
        if (++img_count > num_images_parse) {
            break;
        }

        auto frame_start = high_resolution_clock::now();

        // LOAD IMAGE (not parallelized - simulates camera capture)
        int width = 0, height = 0, channels = 0;
        unsigned char* pixels = stbi_load(
            image_path.string().c_str(),
            &width, &height, &channels, 0
        );

        if (!pixels) {
            cerr << "Failed to load image: " << image_path << "\n";
            continue;
        }

        const int min_height_bound = height * 0.45;
        const int max_height_bound = height * 0.8;

        // ====================================================================
        // PARALLEL SECTION 1: SCENE ANALYSIS
        // Divide image into horizontal strips for parallel brightness analysis
        // ====================================================================
        auto analysis_start = high_resolution_clock::now();
        
        // long long total_brightness = 0;
        // int total_pixels = 0;
        
        // #pragma omp parallel reduction(+:total_brightness, total_pixels)
        // {
        //     int thread_id = omp_get_thread_num();
        //     int num_threads_local = omp_get_num_threads();
            
        //     // Divide height range among threads
        //     int rows_per_thread = (max_height_bound - min_height_bound) / num_threads_local;
        //     int my_start = min_height_bound + thread_id * rows_per_thread;
        //     int my_end = (thread_id == num_threads_local - 1) ? max_height_bound : my_start + rows_per_thread;
            
        //     // Sample every 10th pixel in my assigned rows
        //     for (int j = my_start; j < my_end; j += 10) {
        //         for (int i = 0; i < width; i += 10) {
        //             int idx = (j * width + i) * 3;
        //             int brightness = pixels[idx] + pixels[idx + 1] + pixels[idx + 2];
        //             total_brightness += brightness;
        //             total_pixels++;
        //         }
        //     }
        // }

        SceneStats scene = analyze_scene_cuda(
            pixels,
            width,
            height,
            min_height_bound,
            max_height_bound
        );

        auto analysis_end = high_resolution_clock::now();
        auto analysis_time = duration_cast<microseconds>(analysis_end - analysis_start).count();
        
        // SceneStats scene;
        // scene.avg_brightness = (float)total_brightness / (float)(total_pixels * 3);
        // scene.is_dark_scene = scene.avg_brightness < 60.0f;
        
        // auto analysis_end = high_resolution_clock::now();
        // auto analysis_time = duration_cast<microseconds>(analysis_end - analysis_start).count();

        cout << "\n[Frame " << img_count << "] " << image_path.filename() << endl;
        cout << "  Scene analysis: " << analysis_time / 1000.0 << " ms" << endl;
        cout << "  Avg brightness: " << scene.avg_brightness 
             << " (dark: " << (scene.is_dark_scene ? "YES" : "NO") << ")" << endl;

        // ====================================================================
        // PARALLEL SECTION 2: PIXEL CLASSIFICATION
        // Divide image into horizontal strips, each thread processes its strip
        // ====================================================================
        auto classification_start = high_resolution_clock::now();
        
        // vector<unsigned char> output(width * height * 3, 0);
        
        // // Pre-allocate point vectors for each thread
        // int num_threads_actual = omp_get_max_threads();
        // vector<vector<Point>> thread_yellow(num_threads_actual);
        // vector<vector<Point>> thread_orange(num_threads_actual);
        // vector<vector<Point>> thread_blue(num_threads_actual);
        
        // #pragma omp parallel
        // {
        //     int thread_id = omp_get_thread_num();
        //     int num_threads_local = omp_get_num_threads();
            
        //     // Divide height range among threads
        //     int rows_per_thread = (max_height_bound - min_height_bound) / num_threads_local;
        //     int my_start = min_height_bound + thread_id * rows_per_thread;
        //     int my_end = (thread_id == num_threads_local - 1) ? max_height_bound : my_start + rows_per_thread;
            
        //     // Process my assigned rows
        //     for (int j = my_start; j < my_end; j++) {
        //         for (int i = 0; i < width; i++) {
        //             int pixel_index = (j * width) + i;
        //             int channel_index = pixel_index * 3;

        //             unsigned char r = pixels[channel_index + 0];
        //             unsigned char g = pixels[channel_index + 1];
        //             unsigned char b = pixels[channel_index + 2];

        //             ConeColor color = classify_pixel(r, g, b, scene.is_dark_scene);

        //             if (color == ConeColor::Yellow) {
        //                 thread_yellow[thread_id].push_back({i, j, ConeColor::Yellow});
        //                 output[channel_index + 0] = 255;
        //                 output[channel_index + 1] = 255;
        //                 output[channel_index + 2] = 0;
        //             } else if (color == ConeColor::Orange) {
        //                 thread_orange[thread_id].push_back({i, j, ConeColor::Orange});
        //                 output[channel_index + 0] = 255;
        //                 output[channel_index + 1] = 120;
        //                 output[channel_index + 2] = 0;
        //             } else if (color == ConeColor::Blue) {
        //                 thread_blue[thread_id].push_back({i, j, ConeColor::Blue});
        //                 output[channel_index + 0] = 0;
        //                 output[channel_index + 1] = 0;
        //                 output[channel_index + 2] = 255;
        //             }
        //         }
        //     }
        // }
        
        // // Merge results from all threads
        // vector<Point> yellow_points, orange_points, blue_points;
        // for (int t = 0; t < num_threads_actual; t++) {
        //     yellow_points.insert(yellow_points.end(), thread_yellow[t].begin(), thread_yellow[t].end());
        //     orange_points.insert(orange_points.end(), thread_orange[t].begin(), thread_orange[t].end());
        //     blue_points.insert(blue_points.end(), thread_blue[t].begin(), thread_blue[t].end());
        // }
        
        // auto classification_end = high_resolution_clock::now();
        // auto classification_time = duration_cast<microseconds>(classification_end - classification_start).count();
        
        vector<unsigned char> output(width * height * 3, 0);
        vector<Point> yellow_points, orange_points, blue_points;

        classify_pixels_cuda(
            pixels,
            output.data(),
            yellow_points,
            orange_points,
            blue_points,
            width,
            height,
            min_height_bound,
            max_height_bound,
            scene.is_dark_scene
        );

        auto classification_end = high_resolution_clock::now();
        auto classification_time = duration_cast<microseconds>(
            classification_end - classification_start
        ).count();

        cout << "  Pixel classification: " << classification_time / 1000.0 << " ms" << endl;
        cout << "  Colored pixels - Yellow: " << yellow_points.size() 
             << ", Orange: " << orange_points.size()
             << ", Blue: " << blue_points.size() << endl;

        // ====================================================================
        // PARALLEL SECTION 3: CLUSTERING
        // Can parallelize the three color channels independently
        // ====================================================================
        auto clustering_start = high_resolution_clock::now();
        
        // Adaptive DBSCAN parameters based on scene
        float eps = scene.is_dark_scene ? 12.0f : 7.0f;
        int min_pts = scene.is_dark_scene ? 10 : 30;

        vector<BoundingBox> yellow_boxes, orange_boxes, blue_boxes;
        
        // Cluster each color in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                yellow_boxes = cluster_cones(yellow_points, eps, min_pts);
                yellow_boxes = merge_overlapping_boxes(yellow_boxes);
            }
            
            #pragma omp section
            {
                orange_boxes = cluster_cones(orange_points, eps, min_pts);
                orange_boxes = merge_overlapping_boxes(orange_boxes);
            }
            
            #pragma omp section
            {
                blue_boxes = cluster_cones(blue_points, eps, min_pts);
                blue_boxes = merge_overlapping_boxes(blue_boxes);
            }
        }
        
        auto clustering_end = high_resolution_clock::now();
        auto clustering_time = duration_cast<microseconds>(clustering_end - clustering_start).count();

        cout << "  Clustering: " << clustering_time / 1000.0 << " ms" << endl;
        cout << "  Detected cones - Yellow: " << yellow_boxes.size() 
             << ", Orange: " << orange_boxes.size()
             << ", Blue: " << blue_boxes.size() << endl;

        // ====================================================================
        // PARALLEL SECTION 4: BOUNDING BOX DRAWING
        // Draw boxes in parallel (independent operations)
        // ====================================================================
        auto drawing_start = high_resolution_clock::now();
        
        // Draw all yellow boxes
        #pragma omp parallel for if(yellow_boxes.size() > 5)
        for (size_t i = 0; i < yellow_boxes.size(); i++) {
            draw_rectangle(output, width, height, yellow_boxes[i], 255, 255, 0, 3);
        }
        
        // Draw all orange boxes
        #pragma omp parallel for if(orange_boxes.size() > 5)
        for (size_t i = 0; i < orange_boxes.size(); i++) {
            draw_rectangle(output, width, height, orange_boxes[i], 255, 120, 0, 3);
        }
        
        // Draw all blue boxes
        #pragma omp parallel for if(blue_boxes.size() > 5)
        for (size_t i = 0; i < blue_boxes.size(); i++) {
            draw_rectangle(output, width, height, blue_boxes[i], 0, 100, 255, 3);
        }
        
        auto drawing_end = high_resolution_clock::now();
        auto drawing_time = duration_cast<microseconds>(drawing_end - drawing_start).count();

        cout << "  Drawing boxes: " << drawing_time / 1000.0 << " ms" << endl;

        // SAVE OUTPUT (not parallelized - simulates transmission)
        fs::path output_file = filtered_img_path / image_path.filename();
        output_file.replace_extension(".png");

        stbi_write_png(
            output_file.string().c_str(),
            width, height, 3,
            output.data(),
            width * 3
        );

        stbi_image_free(pixels);
        
        auto frame_end = high_resolution_clock::now();
        auto frame_time = duration_cast<microseconds>(frame_end - frame_start).count();
        
        cout << "  TOTAL FRAME TIME: " << frame_time / 1000.0 << " ms";
        cout << " (" << 1000000.0 / frame_time << " FPS)" << endl;
        cout << "  Output: " << output_file << endl;
    }

    return 0;
}