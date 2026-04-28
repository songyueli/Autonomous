/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: Latest
 * Class: ECE 4122
 * 
 * Description: Optimized DBSCAN with correct OpenMP parallelization
*/

#include "ConeFilter.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

using namespace std;

// Calculate Euclidean distance between two points
inline float distance(const Point& a, const Point& b) 
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Find all neighbors within eps distance - OPTIMIZED
vector<int> region_query(const vector<Point>& points, int point_idx, float eps) 
{
    vector<int> neighbors;
    const int n = points.size();
    
    // Pre-allocate approximate size to reduce reallocations
    neighbors.reserve(n / 10);
    
    // For small datasets, parallelization overhead is not worth it
    if (n < 1000) 
    {
        for (int i = 0; i < n; i++) 
        {
            if (distance(points[point_idx], points[i]) <= eps) 
            {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }
    
    // For large datasets, use parallel reduction pattern
    // Each thread collects its own neighbors, then we merge
    int num_threads = omp_get_max_threads();
    vector<vector<int>> thread_neighbors(num_threads);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_neighbors[tid].reserve(n / (10 * num_threads));
        
        #pragma omp for nowait
        for (int i = 0; i < n; i++) 
        {
            if (distance(points[point_idx], points[i]) <= eps) 
            {
                thread_neighbors[tid].push_back(i);
            }
        }
    }
    
    // Merge all thread results
    for (int t = 0; t < num_threads; t++) 
    {
        neighbors.insert(neighbors.end(), 
                        thread_neighbors[t].begin(), 
                        thread_neighbors[t].end());
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
) 
{
    vector<int> seeds = region_query(points, point_idx, eps);
    
    if (seeds.size() < min_pts) 
    {
        labels[point_idx] = -1; // Mark as noise
        return;
    }
    
    // All points in seeds are part of this cluster
    // CANNOT parallelize this - race condition on labels array
    for (int seed_idx : seeds) 
    {
        labels[seed_idx] = cluster_id;
    }
    
    // Process each seed point
    // CANNOT parallelize outer loop - seeds vector is modified during iteration
    for (size_t i = 0; i < seeds.size(); i++) 
    {
        int current_idx = seeds[i];
        
        if (visited[current_idx]) 
        {
            continue;
        }
        
        visited[current_idx] = true;
        
        vector<int> current_neighbors = region_query(points, current_idx, eps);
        
        if (current_neighbors.size() >= min_pts) 
        {
            // Add new neighbors to seeds
            // CANNOT parallelize - modifying shared seeds vector
            for (int neighbor_idx : current_neighbors) 
            {
                if (labels[neighbor_idx] == 0 || labels[neighbor_idx] == -1) 
                {
                    if (labels[neighbor_idx] == 0) 
                    {
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
) 
{
    int n = points.size();
    vector<int> labels(n, 0);
    vector<bool> visited(n, false);
    int cluster_id = 0;
    
    // DBSCAN main loop - inherently sequential due to data dependencies
    // Each cluster depends on previous clustering decisions
    for (int i = 0; i < n; i++) 
    {
        if (visited[i]) 
        {
            continue;
        }
        
        visited[i] = true;
        cluster_id++;
        expand_cluster(points, labels, i, cluster_id, eps, min_pts, visited);
    }
    
    // ========================================================================
    // OPTIMIZATION: Parallelize bounding box creation
    // This is where the real speedup comes from!
    // ========================================================================
    
    // First, count how many clusters we have and collect points per cluster
    vector<vector<int>> cluster_points(cluster_id + 1);
    
    // Parallelize point collection - each thread handles a range of points
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        
        // Thread-local storage
        vector<vector<int>> local_clusters(cluster_id + 1);
        
        // Divide work among threads
        #pragma omp for nowait
        for (int i = 0; i < n; i++) 
        {
            if (labels[i] > 0) 
            {
                local_clusters[labels[i]].push_back(i);
            }
        }
        
        // Merge local results into global (critical section)
        #pragma omp critical
        {
            for (int c = 1; c <= cluster_id; c++) 
            {
                cluster_points[c].insert(cluster_points[c].end(),
                                        local_clusters[c].begin(),
                                        local_clusters[c].end());
            }
        }
    }
    
    // Now create bounding boxes in parallel - one thread per cluster
    vector<BoundingBox> bounding_boxes;
    bounding_boxes.reserve(cluster_id);
    
    // Use a vector to collect boxes from each thread
    vector<vector<BoundingBox>> thread_boxes(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // Each iteration processes one cluster independently
        #pragma omp for schedule(dynamic)
        for (int cluster = 1; cluster <= cluster_id; cluster++) 
        {
            const auto& point_indices = cluster_points[cluster];
            
            if (point_indices.empty()) 
            {
                continue;
            }
            
            BoundingBox box;
            box.x_min = numeric_limits<int>::max();
            box.y_min = numeric_limits<int>::max();
            box.x_max = numeric_limits<int>::min();
            box.y_max = numeric_limits<int>::min();
            box.pixel_count = point_indices.size();
            
            // Find bounding box extents
            for (int idx : point_indices) 
            {
                box.x_min = min(box.x_min, points[idx].x);
                box.y_min = min(box.y_min, points[idx].y);
                box.x_max = max(box.x_max, points[idx].x);
                box.y_max = max(box.y_max, points[idx].y);
                box.color = points[idx].color;
            }
            
            // Calculate box dimensions
            int width = box.x_max - box.x_min;
            int height = box.y_max - box.y_min;
            float aspect_ratio = (float)width / (float)(height + 1);
            int area = width * height;
            
            // Apply filters
            if (box.pixel_count < max(5, min_pts - 3)) continue;
            if (width > 280 || height > 200) continue;
            if (area < 12) continue;
            
            float density = (float)box.pixel_count / (float)(area + 1);
            if (density < 0.3f) continue;
            
            // Add small padding
            box.x_min = max(0, box.x_min - 2);
            box.y_min = max(0, box.y_min - 2);
            box.x_max += 2;
            box.y_max += 2;
            
            // Add to thread-local box collection
            thread_boxes[tid].push_back(box);
        }
    }
    
    // Merge all thread results
    for (const auto& boxes : thread_boxes) 
    {
        bounding_boxes.insert(bounding_boxes.end(), boxes.begin(), boxes.end());
    }
    
    return bounding_boxes;
}


// ============================================================================
// ALTERNATIVE: Spatial partitioning for DBSCAN parallelization
// This is more complex but can parallelize the clustering itself
// ============================================================================

// Divide points into spatial grid cells
struct SpatialGrid 
{
    int cell_size;
    int grid_width, grid_height;
    int img_width, img_height;
    vector<vector<vector<int>>> cells;  // cells[y][x] = list of point indices
    
    SpatialGrid(const vector<Point>& points, int img_w, int img_h, float eps) 
    {
        cell_size = (int)(eps * 2);  // Cell size should be at least 2*eps
        img_width = img_w;
        img_height = img_h;
        grid_width = (img_w + cell_size - 1) / cell_size;
        grid_height = (img_h + cell_size - 1) / cell_size;
        
        cells.resize(grid_height, vector<vector<int>>(grid_width));
        
        // Assign points to cells
        for (int i = 0; i < points.size(); i++) 
        {
            int cell_x = points[i].x / cell_size;
            int cell_y = points[i].y / cell_size;
            if (cell_x >= 0 && cell_x < grid_width && 
                cell_y >= 0 && cell_y < grid_height) 
                {
                cells[cell_y][cell_x].push_back(i);
            }
        }
    }
    
    // Get neighboring cells for range query
    vector<int> get_neighbors_in_range(const vector<Point>& points, 
                                       int point_idx, float eps) 
    {
        const Point& p = points[point_idx];
        int cx = p.x / cell_size;
        int cy = p.y / cell_size;
        
        vector<int> neighbors;
        
        // Check this cell and 8 surrounding cells
        for (int dy = -1; dy <= 1; dy++) 
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = cx + dx;
                int ny = cy + dy;
                
                if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) 
                {
                    continue;
                }
                
                for (int idx : cells[ny][nx]) 
                {
                    if (distance(points[point_idx], points[idx]) <= eps) 
                    {
                        neighbors.push_back(idx);
                    }
                }
            }
        }
        
        return neighbors;
    }
};

// Optimized DBSCAN using spatial partitioning
vector<BoundingBox> cluster_cones_optimized(
    const vector<Point>& points,
    float eps,
    int min_pts,
    int img_width,
    int img_height
) 
{
    int n = points.size();
    
    // Build spatial grid for faster neighbor queries
    SpatialGrid grid(points, img_width, img_height, eps);
    
    vector<int> labels(n, 0);
    vector<bool> visited(n, false);
    int cluster_id = 0;
    
    // DBSCAN main loop - still sequential but with faster neighbor queries
    for (int i = 0; i < n; i++)
    {
        if (visited[i]) continue;
        
        visited[i] = true;
        
        vector<int> seeds = grid.get_neighbors_in_range(points, i, eps);
        
        if (seeds.size() < min_pts) 
        {
            labels[i] = -1;
            continue;
        }
        
        cluster_id++;
        
        for (int seed_idx : seeds) 
        {
            labels[seed_idx] = cluster_id;
        }
        
        for (size_t j = 0; j < seeds.size(); j++) 
        {
            int current_idx = seeds[j];
            
            if (visited[current_idx]) continue;
            
            visited[current_idx] = true;
            
            vector<int> current_neighbors = grid.get_neighbors_in_range(
                points, current_idx, eps);
            
            if (current_neighbors.size() >= min_pts) 
            {
                for (int neighbor_idx : current_neighbors) 
                {
                    if (labels[neighbor_idx] == 0 || labels[neighbor_idx] == -1) 
                    {
                        if (labels[neighbor_idx] == 0) 
                        {
                            seeds.push_back(neighbor_idx);
                        }
                        labels[neighbor_idx] = cluster_id;
                    }
                }
            }
        }
    }
    
    // Use the same parallelized bounding box creation as above
    vector<vector<int>> cluster_points(cluster_id + 1);
    
    #pragma omp parallel
    {
        vector<vector<int>> local_clusters(cluster_id + 1);
        
        #pragma omp for nowait
        for (int i = 0; i < n; i++) 
        {
            if (labels[i] > 0) 
            {
                local_clusters[labels[i]].push_back(i);
            }
        }
        
        #pragma omp critical
        {
            for (int c = 1; c <= cluster_id; c++) 
            {
                cluster_points[c].insert(cluster_points[c].end(),
                                        local_clusters[c].begin(),
                                        local_clusters[c].end());
            }
        }
    }
    
    vector<BoundingBox> bounding_boxes;
    vector<vector<BoundingBox>> thread_boxes(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic)
        for (int cluster = 1; cluster <= cluster_id; cluster++) 
        {
            const auto& point_indices = cluster_points[cluster];
            if (point_indices.empty()) continue;
            
            BoundingBox box;
            box.x_min = numeric_limits<int>::max();
            box.y_min = numeric_limits<int>::max();
            box.x_max = numeric_limits<int>::min();
            box.y_max = numeric_limits<int>::min();
            box.pixel_count = point_indices.size();
            
            for (int idx : point_indices) 
            {
                box.x_min = min(box.x_min, points[idx].x);
                box.y_min = min(box.y_min, points[idx].y);
                box.x_max = max(box.x_max, points[idx].x);
                box.y_max = max(box.y_max, points[idx].y);
                box.color = points[idx].color;
            }
            
            int width = box.x_max - box.x_min;
            int height = box.y_max - box.y_min;
            int area = width * height;
            
            if (box.pixel_count < max(5, min_pts - 3)) continue;
            if (width > 280 || height > 200) continue;
            if (area < 12) continue;
            
            float density = (float)box.pixel_count / (float)(area + 1);
            if (density < 0.3f) continue;
            
            box.x_min = max(0, box.x_min - 2);
            box.y_min = max(0, box.y_min - 2);
            box.x_max += 2;
            box.y_max += 2;
            
            thread_boxes[tid].push_back(box);
        }
    }
    
    for (const auto& boxes : thread_boxes) 
    {
        bounding_boxes.insert(bounding_boxes.end(), boxes.begin(), boxes.end());
    }
    
    return bounding_boxes;
}