#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>
#include <vector>

#include "draw.h"
#include "calculate.h"

using namespace cv;
using namespace std;

int result_cnt = 0;

void draw(Mat ori_img, Mat tar_img) {

	Mat concate_result;
	hconcat(ori_img, tar_img, concate_result);
	imshow("concate_result", concate_result);
	waitKey(0);
}

void draw_img(Mat _ori_img, Mat _tar_img, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts) {
	// draw correspondence with random color
	Mat ori_img, tar_img;
	_ori_img.copyTo(ori_img);
	_tar_img.copyTo(tar_img);

	//for (int i = 0; i < tiny_src_pts.size(); i++) {
	//	Point ori_circle(tiny_src_pts[i].x, tiny_src_pts[i].y);
	//	Point tar_circle(tiny_dst_pts[i].x, tiny_dst_pts[i].y);
	//	Scalar color(0, 0, 255);
	//	circle(ori_img, ori_circle, 1, color, 3);
	//	circle(tar_img, tar_circle, 1, color, 3);

	//}
	// 
	Mat concate_result;
	hconcat(ori_img, tar_img, concate_result);
	int width = ori_img.size().width;
	for (int i = 0; i < tiny_src_pts.size(); i++) {
		Point ori_circle(tiny_src_pts[i].x, tiny_src_pts[i].y);
		Point tar_circle(tiny_dst_pts[i].x + width, tiny_dst_pts[i].y);
		Scalar color(rand() % 256, rand() % 256, rand() % 256);
		line(concate_result, ori_circle, tar_circle, color, 1);
	}

	//imshow("concate_result", concate_result);
	//waitKey(0);

	string save_path = "C:\\CC\\Exp\\CC_Ccode_img\\CC_temp.jpg";
	imwrite(save_path, concate_result);
	cout << "save image in " << save_path << endl;

	// draw outlier correspondence with highlight circle
	//Mat ori_img, tar_img;
	//_ori_img.copyTo(ori_img);
	//_tar_img.copyTo(tar_img);

	//for (int i = 0; i < tiny_src_pts.size(); i++) {
	//	Point ori_circle(tiny_src_pts[i].x, tiny_src_pts[i].y);
	//	Point tar_circle(tiny_dst_pts[i].x, tiny_dst_pts[i].y);
	//	Scalar color(0, 0, 255);
	//	circle(ori_img, ori_circle, 1, color, 3);
	//	circle(tar_img, tar_circle, 1, color, 3);

	//}

	//Mat concate_result;
	//hconcat(ori_img, tar_img, concate_result);
	//int width = ori_img.size().width;
	//for (int i = 0; i < tiny_src_pts.size(); i++) {
	//	Point ori_circle(tiny_src_pts[i].x, tiny_src_pts[i].y);
	//	Point tar_circle(tiny_dst_pts[i].x + width, tiny_dst_pts[i].y);
	//	Scalar color(rand() % 256, rand() % 256, rand() % 256);
	//	//Scalar color(250, 92, 208);
	//	line(concate_result, ori_circle, tar_circle, color, 1);
	//}

	//imshow("concate_result", concate_result);
	//waitKey(0);

	//imwrite("D:\\Chris\\Exp\\CC\\YC\\GH-COOSAC-main\\COOSAC\\Airport\\outlier.jpg", concate_result);
}

void draw_result(Mat _ori_img, Mat _tar_img, vector<Point2f>& src_pts, vector<Point2f>& dst_pts, vector<int> mask) {
	vector<Point2f> src_choose, dst_choose;
	for (int i = 0; i < mask.size(); i++) {
		if (mask[i] == 1) {
			src_choose.push_back(src_pts[i]);
			dst_choose.push_back(dst_pts[i]);
		}
	}
	draw_img(_ori_img, _tar_img, src_choose, dst_choose);
	return;
}

void draw_output(Mat _ori_img, Mat _tar_img, vector<Point2f>& src_pts, vector<Point2f>& dst_pts, vector<int> gt, vector<int> output) {
	Mat ori_img, tar_img;
	_ori_img.copyTo(ori_img);
	_tar_img.copyTo(tar_img);

	int divider_width = 10;
	Mat divider = Mat::zeros(Size(ori_img.cols, divider_width), CV_32F);
	Mat divided_ori;
	Mat concate_result;
	bool stop = false;
	int FN = 0;
	int FP = 0;
	while (!stop) {
		FN = 0;
		FP = 0;
		//hconcat(ori_img, divider, divided_ori);
		//hconcat(divided_ori, tar_img, concate_result);
		hconcat(ori_img, tar_img, concate_result);
		int width = ori_img.size().width;
		//int width = ori_img.size().width + divider_width;

		// random select 100 correspondences
		int corr_size = gt.size();
		int choose_size = 100;
		srand(time(NULL));

		vector<int> choose_idx(choose_size, 0);
		for (int i = 0; i < choose_size; i++) {
			int tmp = rand() % corr_size;
			// avoid true negative
			while(gt[tmp] == 0 && output[tmp] == 0)
				tmp = rand() % corr_size;
			choose_idx[i] = tmp;
		}
		for (int i = 0; i < choose_idx.size(); i++) {
			int corr_idx = choose_idx[i];
			Scalar line_color;
			// set line color
			if (gt[corr_idx] == 1) {
				if (output[corr_idx] == 1) {
					line_color = Scalar(255, 0, 0); // TP: blue
				}
				else {
					line_color = Scalar(0, 255, 0); // FN: green
					FN++;
				}
			}
			else {
				if (output[corr_idx] == 1) {
					line_color = Scalar(0, 0, 255); // FP: red
					FP++;
				}
				else {
					line_color = Scalar(255, 255, 255); // TN: white
				}
			}

			line(concate_result, src_pts[corr_idx], dst_pts[corr_idx] + Point2f(width, 0), line_color, 2);
		}
		imshow("concate_result", concate_result);
		cout << "FN = " << FN << endl;
		cout << "FP = " << FP << endl;
		//cout << "Inlier rate: " << (double)count(gt.begin(), gt.end(), 1) / gt.size() << endl;;
		char key = (char)waitKey(0);
		if (key == 27) {
			string save_path = "C:\\CC\\Exp\\Fig\\" + to_string(result_cnt) + ".jpg";
			imwrite(save_path, concate_result);
			cout << "save image in " << save_path << endl;
			result_cnt++;
			stop = true;
		}
		else if (key == 8) {
			cout << "skip image\n";
			stop = true;
		}
		else if (key == 32) {
			continue;
		}


	}

	return;
}
