#ifndef CONE_FILTER_CUDA_H
#define CONE_FILTER_CUDA_H

#include "ConeFilter.h"
#include <vector>

// Host-callable CUDA functions
SceneStats analyze_scene_cuda(const unsigned char* pixels, int width, int height,
                              int min_height, int max_height);

void classify_pixels_cuda(const unsigned char* input,
                         unsigned char* output,
                         std::vector<Point>& yellow_points,
                         std::vector<Point>& orange_points,
                         std::vector<Point>& blue_points,
                         int width, int height,
                         int min_height, int max_height,
                         bool is_dark_scene);

#endif // CONE_FILTER_CUDA_H
