/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: Latest
 * Class: ECE 4122
 * 
 * Description: This is the main program
*/

#include "ConeFilter.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// Calculate Euclidean distance between two points
float distance(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Find all neighbors within eps distance
vector<int> region_query(const vector<Point>& points, int point_idx, float eps) {
    vector<int> neighbors;
    
    for (int i = 0; i < points.size(); i++) {
        if (distance(points[point_idx], points[i]) <= eps) {
            neighbors.push_back(i);
        }
    }
    
    return neighbors;
}

// Expand cluster from seed point
void expand_cluster(
    const vector<Point>& points,
    vector<int>& labels,
    int point_idx,
    int cluster_id,
    float eps,
    int min_pts,
    vector<bool>& visited
) {
    vector<int> seeds = region_query(points, point_idx, eps);
    
    if (seeds.size() < min_pts) {
        labels[point_idx] = -1; // Mark as noise
        return;
    }
    
    // All points in seeds are part of this cluster
    for (int seed_idx : seeds) {
        labels[seed_idx] = cluster_id;
    }
    
    // Process each seed point
    for (size_t i = 0; i < seeds.size(); i++) {
        int current_idx = seeds[i];
        
        if (visited[current_idx]) {
            continue;
        }
        
        visited[current_idx] = true;
        
        vector<int> current_neighbors = region_query(points, current_idx, eps);
        
        if (current_neighbors.size() >= min_pts) {
            // Add new neighbors to seeds
            for (int neighbor_idx : current_neighbors) {
                if (labels[neighbor_idx] == 0 || labels[neighbor_idx] == -1) {
                    if (labels[neighbor_idx] == 0) {
                        seeds.push_back(neighbor_idx);
                    }
                    labels[neighbor_idx] = cluster_id;
                }
            }
        }
    }
}

vector<BoundingBox> cluster_cones(
    const vector<Point>& points,
    float eps,
    int min_pts
) {
    int n = points.size();
    vector<int> labels(n, 0);
    vector<bool> visited(n, false);
    int cluster_id = 0;
    
    for (int i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }
        
        visited[i] = true;
        cluster_id++;
        expand_cluster(points, labels, i, cluster_id, eps, min_pts, visited);
    }
    
    // Convert clusters to bounding boxes with geometric filtering
    vector<BoundingBox> bounding_boxes;
    
    for (int cluster = 1; cluster <= cluster_id; cluster++) {
        BoundingBox box;
        box.x_min = numeric_limits<int>::max();
        box.y_min = numeric_limits<int>::max();
        box.x_max = numeric_limits<int>::min();
        box.y_max = numeric_limits<int>::min();
        box.pixel_count = 0;
        
        bool has_points = false;
        
        for (int i = 0; i < n; i++) {
            if (labels[i] == cluster) {
                has_points = true;
                box.x_min = min(box.x_min, points[i].x);
                box.y_min = min(box.y_min, points[i].y);
                box.x_max = max(box.x_max, points[i].x);
                box.y_max = max(box.y_max, points[i].y);
                box.color = points[i].color;
                box.pixel_count++;
            }
        }
        
        if (!has_points) continue;
        
        // Calculate box dimensions
        int width = box.x_max - box.x_min;
        int height = box.y_max - box.y_min;
        float aspect_ratio = (float)width / (float)(height + 1);
        int area = width * height;
        
        // FILTER 1: Lower minimum pixel count
        if (box.pixel_count < max(5, min_pts - 3)) {  // At least 5 pixels, or min_pts-3
            continue;
        }

        // FILTER 2: Maximum dimensions
        if (width > 280 || height > 200) {
            continue;
        }

        // FILTER 3: Aspect ratio - even wider range
        if (aspect_ratio < 0.2f || aspect_ratio > 4.0f) {
            continue;
        }

        // FILTER 4: Density check - much more lenient
        float density = (float)box.pixel_count / (float)(area + 1);
        if (density < 0.25f) { 
            continue;
        }

        // FILTER 5: Minimum area - smaller minimum
        if (area < 15) {  // Was 20
            continue;
        }
        
        // Add small padding
        box.x_min = max(0, box.x_min - 2);
        box.y_min = max(0, box.y_min - 2);
        box.x_max += 2;
        box.y_max += 2;
        
        bounding_boxes.push_back(box);
    }
    
    return bounding_boxes;
}