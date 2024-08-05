#pragma once
#ifndef DRAW_H
#define DRAW_H
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


void draw(Mat ori_img, Mat tar_img);

void draw_img(Mat ori_img, Mat tar_img, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts);

void draw_result(Mat _ori_img, Mat _tar_img, vector<Point2f>& src_pts, vector<Point2f>& dst_pts, vector<int> mask);

void draw_output(Mat _ori_img, Mat _tar_img, vector<Point2f>& src_pts, vector<Point2f>& dst_pts, vector<int> gt, vector<int> output);
#endif // !DRAW_H
