/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: Latest
 * Class: ECE 4122
 * 
 * Description: This is the main program
*/

#ifndef CONEFILTER_H
#define CONEFILTER_H

#include <vector>
#include <cmath>

using namespace std;

enum class ConeColor {
    None,
    Yellow,
    Orange,
    Blue
};

struct HSV {
    float h;  // 0-360
    float s;  // 0-1
    float v;  // 0-1
};

struct Point {
    int x;
    int y;
    ConeColor color;
};

struct BoundingBox {
    int x_min;
    int y_min;
    int x_max;
    int y_max;
    ConeColor color;
    int pixel_count;
};

struct SceneStats {
    float avg_brightness;
    bool is_dark_scene;
};

SceneStats analyze_scene(const unsigned char* pixels, int width, int height,
                         int min_height, int max_height);

HSV rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b);
ConeColor classify_pixel(unsigned char r, unsigned char g, unsigned char b, 
                        bool is_dark_scene);

// DBSCAN clustering
vector<BoundingBox> cluster_cones(
    const vector<Point>& points,
    float eps,
    int min_pts
);

vector<BoundingBox> merge_overlapping_boxes(vector<BoundingBox>& boxes);


#endif