#include "ConeFilterCUDA.cuh"
#include <cuda_runtime.h>
#include <cmath>

// Device function for RGB to HSV conversion
__device__ HSV rgb_to_hsv_device(unsigned char r, unsigned char g, unsigned char b) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;
    
    float maxc = fmaxf(rf, fmaxf(gf, bf));
    float minc = fminf(rf, fminf(gf, bf));
    float delta = maxc - minc;
    
    HSV hsv;
    hsv.v = maxc;
    hsv.s = (maxc > 0.0f) ? (delta / maxc) : 0.0f;
    
    if (delta == 0.0f) {
        hsv.h = 0.0f;
    } else if (maxc == rf) {
        hsv.h = 60.0f * fmodf((gf - bf) / delta, 6.0f);
    } else if (maxc == gf) {
        hsv.h = 60.0f * ((bf - rf) / delta + 2.0f);
    } else {
        hsv.h = 60.0f * ((rf - gf) / delta + 4.0f);
    }
    
    if (hsv.h < 0.0f) hsv.h += 360.0f;
    
    return hsv;
}

// Device function for pixel classification
__device__ ConeColor classify_pixel_device(unsigned char r, unsigned char g, unsigned char b, 
                                           bool is_dark_scene) {
    int brightness = r + g + b;
    float rg_ratio = (float)r / ((float)g + 1.0f);
    float br_ratio = (float)b / ((float)r + 1.0f);
    float bg_ratio = (float)b / ((float)g + 1.0f);

    // Warm color detection
    HSV hsv = rgb_to_hsv_device(r, g, b);
    bool is_bright_pixel = hsv.v > 0.75f;

    if (brightness < 40) {
        return ConeColor::None;
    }
    
    // Blue detection
    bool blue_by_ratio = (
        b >= 20 &&
        br_ratio > 1.15f &&
        bg_ratio > 1.15f &&
        hsv.v > 0.12f
    );
    
    if (blue_by_ratio) {
        return ConeColor::Blue;
    }

    if (hsv.v < (is_dark_scene ? 0.10f : 0.18f)) {
        // Only consider very dark shades of blue 
        // if (b > 30 && 
        //     hsv.h > 200.0f &&
        //     hsv.s > 0.12f &&
        //     br_ratio > 0.8f && 
        //     br_ratio > 0.8f)
        // {
        //     return ConeColor::Blue;
        // }

        return ConeColor::None;
    }
    
    if (is_bright_pixel && 
        rg_ratio > 0.9f &&
        r > b * 1.15f &&
        g > b * 1.15f)
    {
        return ConeColor::Yellow;
    }

    float sat_threshold = is_dark_scene ? 
        (is_bright_pixel ? 0.18f : 0.25f) : 
        (is_bright_pixel ? 0.22f : 0.35f);
    
    if (hsv.s < sat_threshold) {
        return ConeColor::None;
    }
    
    if (b > r || b > g) {
        return ConeColor::None;
    }
    
    if (hsv.h < 25.0f) {
        return ConeColor::Orange;
    }
    
    if (hsv.h >= 30.0f && hsv.h <= 70.0f &&
        hsv.v > 0.07f &&
        hsv.s > 0.11f &&
        r > 35 &&
        g > 30 &&
        b < 120 &&
        r > b * 1.25f &&
        g > b * 1.15f &&
        rg_ratio < 1.65f && rg_ratio > 1.1f) {
        return ConeColor::Yellow;
    }
    
    if (hsv.h >= 28.0f && hsv.h < 42.0f) {
        if (rg_ratio >= 1.4f)
            return ConeColor::Orange;
        else
            return ConeColor::Yellow;
    }
    
    return ConeColor::None;
}

// CUDA kernel for scene analysis
__global__ void analyze_scene_kernel(const unsigned char* pixels, 
                                     int width, int height,
                                     int min_height, int max_height,
                                     unsigned long long* brightness_sum,
                                     unsigned int* pixel_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Sample every 10th pixel
    if (x >= width || y < min_height || y >= max_height) return;
    if (x % 10 != 0 || y % 10 != 0) return;
    
    int idx = (y * width + x) * 3;
    int brightness = pixels[idx] + pixels[idx + 1] + pixels[idx + 2];
    
    atomicAdd(brightness_sum, (unsigned long long)brightness);
    atomicAdd(pixel_count, 1);
}

// CUDA kernel for pixel classification
__global__ void classify_pixels_kernel(const unsigned char* input,
                                       unsigned char* output,
                                       unsigned char* color_map,
                                       int width, int height,
                                       int min_height, int max_height,
                                       bool is_dark_scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y < min_height || y >= max_height) return;
    
    int idx = (y * width + x) * 3;
    
    unsigned char r = input[idx + 0];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];
    
    ConeColor color = classify_pixel_device(r, g, b, is_dark_scene);
    
    // Store color classification in separate map
    color_map[y * width + x] = static_cast<unsigned char>(color);
    
    // Set output visualization
    if (color == ConeColor::Yellow) {
        output[idx + 0] = 255;
        output[idx + 1] = 255;
        output[idx + 2] = 0;
    } else if (color == ConeColor::Orange) {
        output[idx + 0] = 255;
        output[idx + 1] = 120;
        output[idx + 2] = 0;
    } else if (color == ConeColor::Blue) {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 255;
    } else {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
    }
}

// Host function to analyze scene using CUDA
SceneStats analyze_scene_cuda(const unsigned char* pixels, int width, int height,
                              int min_height, int max_height) {
    // Allocate device memory
    unsigned char* d_pixels;
    unsigned long long* d_brightness_sum;
    unsigned int* d_pixel_count;
    
    size_t image_size = width * height * 3;
    
    cudaMalloc(&d_pixels, image_size);
    cudaMalloc(&d_brightness_sum, sizeof(unsigned long long));
    cudaMalloc(&d_pixel_count, sizeof(unsigned int));
    
    // Initialize counters to zero
    cudaMemset(d_brightness_sum, 0, sizeof(unsigned long long));
    cudaMemset(d_pixel_count, 0, sizeof(unsigned int));
    
    // Copy image to device
    cudaMemcpy(d_pixels, pixels, image_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    analyze_scene_kernel<<<gridSize, blockSize>>>(
        d_pixels, width, height, min_height, max_height,
        d_brightness_sum, d_pixel_count
    );
    
    // Copy results back
    unsigned long long brightness_sum;
    unsigned int pixel_count;
    cudaMemcpy(&brightness_sum, d_brightness_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pixel_count, d_pixel_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_brightness_sum);
    cudaFree(d_pixel_count);
    
    SceneStats stats;
    stats.avg_brightness = (float)brightness_sum / (float)(pixel_count * 3);
    stats.is_dark_scene = stats.avg_brightness < 60.0f;
    
    return stats;
}

// Host function to classify all pixels using CUDA
void classify_pixels_cuda(const unsigned char* input,
                         unsigned char* output,
                         std::vector<Point>& yellow_points,
                         std::vector<Point>& orange_points,
                         std::vector<Point>& blue_points,
                         int width, int height,
                         int min_height, int max_height,
                         bool is_dark_scene) {
    // Allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    unsigned char* d_color_map;
    
    size_t image_size = width * height * 3;
    size_t map_size = width * height;
    
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMalloc(&d_color_map, map_size);
    
    // Copy input to device
    cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, image_size);
    cudaMemset(d_color_map, 0, map_size);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    classify_pixels_kernel<<<gridSize, blockSize>>>(
        d_input, d_output, d_color_map,
        width, height, min_height, max_height,
        is_dark_scene
    );
    
    // Copy results back
    cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost);
    
    unsigned char* h_color_map = new unsigned char[map_size];
    cudaMemcpy(h_color_map, d_color_map, map_size, cudaMemcpyDeviceToHost);
    
    // Extract points from color map
    yellow_points.clear();
    orange_points.clear();
    blue_points.clear();
    
    for (int y = min_height; y < max_height; y++) {
        for (int x = 0; x < width; x++) {
            ConeColor color = static_cast<ConeColor>(h_color_map[y * width + x]);
            
            if (color == ConeColor::Yellow) {
                yellow_points.push_back({x, y, ConeColor::Yellow});
            } else if (color == ConeColor::Orange) {
                orange_points.push_back({x, y, ConeColor::Orange});
            } else if (color == ConeColor::Blue) {
                blue_points.push_back({x, y, ConeColor::Blue});
            }
        }
    }
    
    delete[] h_color_map;
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_color_map);
}
