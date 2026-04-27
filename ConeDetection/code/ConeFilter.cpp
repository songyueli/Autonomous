#include "ConeFilter.h"
#include <algorithm>
#include <cmath>


SceneStats analyze_scene(const unsigned char* pixels, int width, int height,
                        int min_height, int max_height) {
    long long brightness_sum = 0;
    int pixel_count = 0;
    
    // Sample every 10th pixel for speed
    for (int j = min_height; j < max_height; j += 10) {
        for (int i = 0; i < width; i += 10) {
            int idx = (j * width + i) * 3;
            int brightness = pixels[idx] + pixels[idx + 1] + pixels[idx + 2];
            brightness_sum += brightness;
            pixel_count++;
        }
    }
    
    SceneStats stats;
    stats.avg_brightness = (float)brightness_sum / (float)(pixel_count * 3);
    stats.is_dark_scene = stats.avg_brightness < 60.0f; // Threshold for "dark"
    
    return stats;
}


HSV rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;
    
    float maxc = std::max({rf, gf, bf});
    float minc = std::min({rf, gf, bf});
    float delta = maxc - minc;
    
    HSV hsv;
    hsv.v = maxc;
    hsv.s = (maxc > 0.0f) ? (delta / maxc) : 0.0f;
    
    if (delta == 0.0f) {
        hsv.h = 0.0f;
    } else if (maxc == rf) {
        hsv.h = 60.0f * fmod((gf - bf) / delta, 6.0f);
    } else if (maxc == gf) {
        hsv.h = 60.0f * ((bf - rf) / delta + 2.0f);
    } else {
        hsv.h = 60.0f * ((rf - gf) / delta + 4.0f);
    }
    
    if (hsv.h < 0.0f) hsv.h += 360.0f;
    
    return hsv;
}

ConeColor classify_pixel(unsigned char r, unsigned char g, unsigned char b, 
                        bool is_dark_scene) {
    
    int brightness = r + g + b;
    
    // Filter completely black pixels
    if (brightness < 25) {
        return ConeColor::None;
    }
    
    // ========== BLUE CONE DETECTION ==========
    // Blue cones are problematic - they're often dark and desaturated
    // Use MULTIPLE detection strategies
    
    // Strategy 1: RGB ratio method (works in all conditions)
    bool blue_by_ratio = (
        b >= 20 &&                           // Minimum blue channel
        b > r * 1.15f &&                     // Blue exceeds red by 15%
        b > g * 1.15f &&                     // Blue exceeds green by 15%
        brightness > 30 &&                   // Not completely black
        brightness < 400                     // Not white/overexposed
    );
    
    // Strategy 2: HSV method for well-lit blue
    HSV hsv = rgb_to_hsv(r, g, b);
    bool blue_by_hsv = (
        hsv.h >= 190.0f && hsv.h <= 260.0f &&
        hsv.s > 0.15f &&                     // Very lenient saturation
        hsv.v > 0.10f                        // Very lenient value
    );
    
    // Strategy 3: Absolute blue dominance (for very dark blues)
    bool blue_by_absolute = (
        b > r + 5 &&                         // Blue clearly higher than red
        b > g + 5 &&                         // Blue clearly higher than green
        b >= 25 &&                           // Minimum absolute blue
        brightness >= 40                     // Minimum total brightness
    );
    
    // Accept if ANY strategy identifies it as blue
    if (blue_by_ratio) // || blue_by_hsv || blue_by_absolute) 
    {
        return ConeColor::Blue;
    }
    
    // ========== ORANGE/YELLOW DETECTION ==========
    
    // For HSV-based detection
    if (hsv.v < (is_dark_scene ? 0.10f : 0.18f)) {
        return ConeColor::None;
    }
    
    // Warm color hue range
    bool in_warm_range = (hsv.h >= 15.0f && hsv.h <= 70.0f);
    if (!in_warm_range) {
        return ConeColor::None;
    }
    
    // Adaptive saturation threshold
    bool is_bright_pixel = hsv.v > 0.65f;
    float sat_threshold;
    
    if (is_dark_scene) {
        sat_threshold = is_bright_pixel ? 0.18f : 0.25f;
    } else {
        sat_threshold = is_bright_pixel ? 0.22f : 0.35f;
    }
    
    if (hsv.s < sat_threshold) {
        return ConeColor::None;
    }
    
    // Additional RGB check for orange/yellow
    // Orange/yellow should have r > b and g > b
    if (b > r || b > g) {
        return ConeColor::None;
    }
    
    // ORANGE: 15-35° hue
    if (hsv.h >= 15.0f && hsv.h < 35.0f) {
        return ConeColor::Orange;
    }
    
    // YELLOW: 35-70° hue
    if (hsv.h >= 35.0f && hsv.h <= 70.0f) {
        return ConeColor::Yellow;
    }
    
    return ConeColor::None;
}

// Merge overlapping boxes of similar colors
vector<BoundingBox> merge_overlapping_boxes(vector<BoundingBox>& boxes) {
    vector<BoundingBox> merged;
    vector<bool> used(boxes.size(), false);
    
    for (size_t i = 0; i < boxes.size(); i++) {
        if (used[i]) continue;
        
        BoundingBox current = boxes[i];
        used[i] = true;
        
        // Check for vertical overlap with same color
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (used[j] || boxes[j].color != current.color) continue;
            
            // Check if boxes overlap or are very close vertically
            int x_overlap = min(current.x_max, boxes[j].x_max) - 
                           max(current.x_min, boxes[j].x_min);
            int y_gap = max(0, max(current.y_min, boxes[j].y_min) - 
                               min(current.y_max, boxes[j].y_max));
            
            // Merge if significant horizontal overlap and small vertical gap
            if (x_overlap > 10 && y_gap < 25) {
                current.x_min = min(current.x_min, boxes[j].x_min);
                current.y_min = min(current.y_min, boxes[j].y_min);
                current.x_max = max(current.x_max, boxes[j].x_max);
                current.y_max = max(current.y_max, boxes[j].y_max);
                current.pixel_count += boxes[j].pixel_count;
                used[j] = true;
            }
        }
        
        merged.push_back(current);
    }
    
    return merged;
}