/**
 * Author: Song Yue David Li, Adish Mittal
 * Date Modified: Latest
 * Class: ECE 4122
 * 
 * Description: This is the main program
*/

struct ConePoint
{
    float x, y;
    int category_id; 
};


bool is_yellow_cone(unsigned char r, unsigned char g, unsigned char b);

bool is_orange_cone(unsigned char r, unsigned char g, unsigned char b);

bool is_blue_cone(unsigned char r, unsigned char g, unsigned char b);