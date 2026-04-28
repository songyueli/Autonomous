#include <omp.h>

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
) 
{
    // Parallelize rectangle drawing for large boxes
    #pragma omp parallel for if(box.x_max - box.x_min > 100)
    for (int x = box.x_min; x <= box.x_max; x++) 
    {
        if (x < 0 || x >= width) continue;
        
        for (int t = 0; t < thickness; t++) 
        {
            // Top edge
            int y_top = box.y_min + t;
            if (y_top >= 0 && y_top < height) 
            {
                int idx = (y_top * width + x) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
            
            // Bottom edge
            int y_bot = box.y_max - t;
            if (y_bot >= 0 && y_bot < height) 
            {
                int idx = (y_bot * width + x) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
        }
    }
    
    #pragma omp parallel for if(box.y_max - box.y_min > 100)
    for (int y = box.y_min; y <= box.y_max; y++) 
    {
        if (y < 0 || y >= height) continue;
        
        for (int t = 0; t < thickness; t++) 
        {
            // Left edge
            int x_left = box.x_min + t;
            if (x_left >= 0 && x_left < width) 
            {
                int idx = (y * width + x_left) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
            
            // Right edge
            int x_right = box.x_max - t;
            if (x_right >= 0 && x_right < width) 
            {
                int idx = (y * width + x_right) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
        }
    }
}