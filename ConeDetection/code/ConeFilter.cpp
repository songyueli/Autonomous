/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: Latest
 * Class: ECE 4122
 * 
 * Description: This is the main program
*/

#include <cmath>

#include "ConeFilter.h"

using namespace std;

bool is_orange_cone(unsigned char r, unsigned char g, unsigned char b)
{
    int brightness = r + g + b;
    return brightness > 80 &&
           r > 55 && r < 90 &&
           g > 25 && g < 65 &&
           b < 60 &&
           r > g * 1.2 &&
           r > b * 1.1;

    // 74, 39, 42
}

bool is_yellow_cone(unsigned char r, unsigned char g, unsigned char b)
{
    int brightness = r + g + b;

    return brightness > 100 &&
           r > 80 &&
           g > 60 &&
           b < 85 &&
           abs((int)r - (int)g) > 15 && 
           abs((int)r - (int)b) > 20;

    // 51, 42, 30
}

bool is_blue_cone(unsigned char r, unsigned char g, unsigned char b)
{
    int brightness = r + g + b;
    return brightness < 120 && 
           r < 50 && r > 20 &&
           g < 50 && g > 20 &&
           b < 50 && b > 20 &&
           abs((int)r - (int)g) < 20 && 
           abs((int)g - (int)b) < 20;
}

ConeColor classify_pixel(unsigned char r, unsigned char g, unsigned char b)
{
    int brightness = r + g + b;

    if (brightness < 40)
        return ConeColor::None;

    float sum = brightness + 1.0f;

    float rn = r / sum;
    float gn = g / sum;
    float bn = b / sum;

    float yellow = rn + gn - 2.0f * bn;
    float blue = bn - 0.45f * (rn + gn);
    float orange = rn - 0.7f * gn - 0.8f * bn;

    if (blue > 0.08f && b > 35)
        return ConeColor::Blue;

    if (yellow > 0.28f && r > 35 && g > 30)
        return ConeColor::Yellow;

    if (orange > 0.12f && r > g)
        return ConeColor::Orange;

    return ConeColor::None;
}