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
           r > 42 && r < 90 &&
           g > 25 && g < 65 &&
           b < 60 &&
           r > g * 1.2 &&
           r > b * 1.1;
}

bool is_yellow_cone(unsigned char r, unsigned char g, unsigned char b)
{
    int brightness = r + g + b;

    return brightness > 100 &&
           r > 80 &&
           g > 75 &&
           b < 55 &&
           r > b * 1.4 &&
           g > b * 1.15 &&
           abs((int)r - (int)g) < 40;
}

bool is_blue_cone(unsigned char r, unsigned char g, unsigned char b)
{
    int brightness = r + g + b;
    return brightness < 120 && 
           r > 25 && r < 35 &&
           g < 65 && g > 25 &&
           b < 70 && b > 35; // &&
           //b > r * 0.8 &&
           //b > g * 0.8;
}