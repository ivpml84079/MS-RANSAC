#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <fstream>
#include <numeric> 
#include <windows.h>
#include <stack>

#include "method.h"
#include "draw.h"
#include "calculate.h"
#include "ransac.h"
#include "svm.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


int default_ransacReprojThreshold = 5; // 低inlier rate效果較好(RANSAC reprojection error)
//int default_ransacReprojThreshold = 10; // 在高inlier rate效果較好
int default_maxIters = 200000;
double default_confidence = 0.995;

void calculate_result(vector<int> ground_truth, vector<int> est_inlier) {
	// print the recall, precision and F1-score of the given result
	vector<int> mask(ground_truth.size(), 0);
	if (ground_truth.size() != est_inlier.size()) { // est_inlier is the list of index instead of mask
		//cout << "calculate result in index mode\n";
		for (int i = 0; i < est_inlier.size(); i++) {
			mask[est_inlier[i]] = 1;
		}
	}
	else { // est_inlier is already the mask
		//cout << "calculate result in mask mode\n";
		mask = est_inlier;
	}
	auto classification = classification_model(ground_truth, mask);
	double TP = classification.TP; double TN = classification.TN;
	double FP = classification.FP; double FN = classification.FN;
	double recall = TP / (TP + FN);
	double precision = TP / (TP + FP);
	double specificity = TN / (TN + FP);
	double f1_score = 2 * precision * recall / (precision + recall);
	cout << "precision: " << precision << ", recall: " << recall << ", f1-score: " << f1_score << endl;
}


vector<int> method_interval(double& f, vector<double>& first, vector<double>& second, bool isLength) {
	vector<double> pitch_first;
	vector<int> fre_first;
	vector<svm_node> tmp(8);

	if (isLength) {
		cal_pitch(first, len_pitch, pitch_first);
		fre_first = cal_frequency_descending(first, pitch_first, tmp);
	}
	else {
		cal_pitch(first, angle_pitch, pitch_first);
		fre_first = cal_frequency_descending(first, pitch_first, tmp);
	}

	svm_model* model = svm_load_model("C:\\svm\\snm_test\\train_orientation_0729.model");
	int range = int(svm_predict(model, tmp.data()));

	vector<int> save_first;
	vector<double> weight_first;
	save_interval(range, first, pitch_first, fre_first, save_first, weight_first);

	vector<double> filter_second(save_first.size());
	for (int i = 0; i < save_first.size(); i++)
		filter_second[i] = second[save_first[i]];

	vector<double> pitch_second;
	vector<int> fre_second;
	vector<svm_node> tmp2(8);
	if (isLength) {
		cal_pitch(filter_second, angle_pitch, pitch_second);
		fre_second = cal_frequency_descending(filter_second, pitch_second, tmp2);
	}
	else {
		cal_pitch(filter_second, len_pitch, pitch_second);
		fre_second = cal_frequency_descending(filter_second, pitch_second, tmp2);
	}
	//start = clock();
	svm_model* model2 = svm_load_model("C:\\svm\\snm_test\\train_length_0729.model");
	range = int(svm_predict(model2, tmp2.data()));

	vector<int> save_second;
	vector<double> weight_second;

	save_interval(range, filter_second, pitch_second, fre_second, save_second, weight_second);

	double inlier_rate = static_cast<double> (save_second.size()) / second.size();

	f = default_ransacReprojThreshold + inlier_rate * default_ransacReprojThreshold;


	vector<pair<int, double>> save_weight(save_second.size());
	for (int i = 0; i < save_second.size(); i++) {
		save_weight[i].first = save_first[save_second[i]];
		save_weight[i].second = weight_first[save_second[i]] + weight_second[i];
	}

	sort_weight(save_weight);
	/*for (auto i : save_weight) 
		cout << i.first << " " << i.second << endl;*/


	vector<int> save_index(save_weight.size());
	for (int i = 0; i < save_weight.size(); i++) 
		save_index[i] = save_weight[i].first;

	return save_index;
}
vector<int> method_interval_ours(int f, vector<double>& first, vector<double>& second, bool isLength) {
	vector<double> pitch_first;
	vector<int> fre_first;
	if (isLength) {
		cal_pitch(first, len_pitch, pitch_first);
		//fre_first = cal_frequency_descending(first, pitch_first);
	}
	else {
		cal_pitch(first, angle_pitch, pitch_first);
		//fre_first = cal_frequency_descending(first, pitch_first);
	}

	vector<int> save_first;
	vector<double> weight_first;
	save_interval(f, first, pitch_first, fre_first, save_first, weight_first);

	cout << "[DEBUG] save_first size : " << save_first.size() << endl;

	//vector<double> filter_second(save_first.size());
	//for (int i = 0; i < save_first.size(); i++)
	//	filter_second[i] = second[save_first[i]];

	//vector<double> pitch_second;
	//vector<int> fre_second;
	//if (isLength) {
	//	cal_pitch(filter_second, angle_pitch, pitch_second);
	//	fre_second = cal_frequency_descending(filter_second, pitch_second);
	//}
	//else {
	//	cal_pitch(filter_second, len_pitch, pitch_second);
	//	fre_second = cal_frequency_descending(filter_second, pitch_second);
	//	//cout << "check frequency : " << endl;
	//	//for (int i = 0; i < pitch_second.size(); i++) {
	//	//	cout << pitch_second[i] << ": ";
	//	//}
	//	//cout << endl;
	//}

	//vector<int> save_second;
	//vector<double> weight_second;
	//save_interval(f, filter_second, pitch_second, fre_second, save_second, weight_second);

	//cout << "[DEBUG] save_second size : " << save_second.size() << endl;

	vector<pair<int, double>> save_weight(save_first.size());
	for (int i = 0; i < save_first.size(); i++) {
		save_weight[i].first = save_first[i];
		save_weight[i].second = weight_first[i] + weight_first[i];
	}

	sort_weight(save_weight);
	/*for (auto i : save_weight)
		cout << i.first << " " << i.second << endl;*/


	vector<int> save_index(save_weight.size());
	for (int i = 0; i < save_weight.size(); i++)
		save_index[i] = save_weight[i].first;

	return save_index;
}

vector<int> NN_filter(vector<Point2f> original_match_pt, vector<Point2f> target_match_pt, vector<int> index) {
	//time_t start, end;
	//start = clock();
	double dis_threshold = 120;
	int pts_num = index.size();
	vector<vector<int>> nn(pts_num, vector<int>());
	for(int i=0; i< pts_num; i++){
		for (int j = i; j < pts_num; j++) {
			double d_x_src = original_match_pt[index[i]].x - original_match_pt[index[j]].x;
			double d_y_src = original_match_pt[index[i]].y - original_match_pt[index[j]].y;
			double d_src = sqrt(d_x_src * d_x_src + d_y_src * d_y_src);
			double d_x_dst = target_match_pt[index[i]].x - target_match_pt[index[j]].x;
			double d_y_dst = target_match_pt[index[i]].y - target_match_pt[index[j]].y;
			double d_dst = sqrt(d_x_dst * d_x_dst + d_y_dst * d_y_dst);
			if (((d_src + d_dst) / 2) < dis_threshold) {
				nn[i].push_back(j);
				nn[j].push_back(i);
			}
		}
	}
	
	
	// finding connected components
	vector<int> res; // 存最終結果
	vector<bool> visit(pts_num, false);
	vector<int> cc_idx(pts_num, -1);
	vector<int> cc_size;
	stack<int> nn_s;
	int cc_idx_count = 0;
	int total_count = 0; // 計算總共有多少點被分群了
	while (total_count < pts_num) {
		for (int i = 0; i < pts_num; i++) {
			if (!visit[i]) {
				nn_s.push(i);
				break;
			}
		}
		int cc_size_count = 0; // 計算這個CC有多少點
		while (!nn_s.empty()) {
			int cur = nn_s.top(); nn_s.pop();
			if (visit[cur])
				continue;
			total_count++;
			cc_size_count++;
			visit[cur] = true;
			cc_idx[cur] = cc_idx_count;
			for (int i = 0; i < nn[cur].size(); i++)
				nn_s.push(nn[cur][i]);
		}
		cc_size.push_back(cc_size_count);
		cc_idx_count++;
	}
	//end = clock();
	//cout << "NN filter time : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
	cout << "CC size: \n";
	for (int i = 0; i < cc_size.size(); i++) {
		cout << cc_size[i] << ", ";
	}
	cout << endl;
	// find the maximum connected component
	// if other CC have similar size
	int max_cc_idx = distance(cc_size.begin(), 
							   max_element(cc_size.begin(), cc_size.end()));
	int max_cc_size = cc_size[max_cc_idx]; //　最大CC大小
	// include the CC that have similar size with the maximum CC
	double sim_thresh = 0.1;
	for (int i = 0; i < pts_num; i++) {
		//if (cc_size[cc_idx[i]] >= sim_thresh * max_cc_size)
		if (cc_size[cc_idx[i]] > 5)
			res.push_back(index[i]);
	}
	//for (int i = 0; i < pts_num; i++) {
	//	if(cc_idx[i] == max_cc_idx)
	//		res.push_back(index[i]);
	//}
	return res;
}

float IR_predictor(vector<Point2f>& src_pts, vector<Point2f>& dst_pts) {
	// preform KNN search on both original_match_pt and target_match_pt
	// create KDTree for both original_match_pt and target_match_pt
	// and perform KNN search

	// transform pts from vector to Mat
	Mat src_pts_mat = Mat(src_pts.size(), 2, CV_32F);
	Mat dst_pts_mat = Mat(dst_pts.size(), 2, CV_32F);
	for (int i = 0; i < src_pts.size(); i++) {
		src_pts_mat.at<float>(i, 0) = src_pts[i].x;
		src_pts_mat.at<float>(i, 1) = src_pts[i].y;
		dst_pts_mat.at<float>(i, 0) = dst_pts[i].x;
		dst_pts_mat.at<float>(i, 1) = dst_pts[i].y;
	}

	int k = 10;
	flann::Index kdtree_src(src_pts_mat, flann::KDTreeIndexParams(2));
	flann::Index kdtree_dst(dst_pts_mat, flann::KDTreeIndexParams(2));
	Mat indices_src, dists_src;
	Mat indices_dst, dists_dst;
	kdtree_src.knnSearch(src_pts_mat, indices_src, dists_src, k+1, flann::SearchParams(128));
	kdtree_dst.knnSearch(dst_pts_mat, indices_dst, dists_dst, k+1, flann::SearchParams(128));

	
	// calculate the common NN rate
	vector<float> cnn_rate(src_pts.size());
//#pragma omp parallel for
	for (int i = 0; i < src_pts.size(); i++) {
		float count = 0.0;
		// ignore the first element because it is the point itself
		for (int j = 1; j < k+1; j++) {
			for (int l = 1; l < k+1; l++) {
				if (indices_src.at<int>(i, j) == indices_dst.at<int>(i, l))
					count += 1;
			}
		}
		cnn_rate[i] = count / k;
	}

	// calculate the mean and standard deviation of cnn_rate
	float mean_cnn_rate = accumulate(cnn_rate.begin(), cnn_rate.end(), 0.0f) / src_pts.size();
	float var_cnn_rate = 0.0;
	float std_cnn_rate;
	for (float value : cnn_rate) {
		var_cnn_rate += pow(value - mean_cnn_rate, 2);
	}
	var_cnn_rate /= src_pts.size();
	std_cnn_rate = sqrt(var_cnn_rate);

	// predict outlier rate using mean and stardard deviation
	float alpha = 0.2;
	float inlier_rate = mean_cnn_rate + 0.1 * (1.0 - exp(-1*std_cnn_rate / alpha));

	return inlier_rate;
}
vector<int> GH_filter(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt) {
	// Vi
	vector<Point2f> vec(original_match_pt.size());
	for (int i = 0; i < original_match_pt.size(); i++) {
		vec[i] = Point2f((target_match_pt[i].x + 1080 - original_match_pt[i].x), (target_match_pt[i].y - original_match_pt[i].y));
	}

	//cout << "[DEBUG] check length:" << endl;
	// length of Vi
	vector<double> vector_len(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double length = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
		vector_len[i] = length;
		//cout << length << ", ";
	}
	//cout << "\n [DEBUG] check angle:\n";
	// angle of Vi
	vector<double> vector_ang(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double theta = atan2(vec[i].y, vec[i].x);
		theta = abs(theta * 180.0 / CV_PI);
		//theta = theta * 180.0 / CV_PI; // 效果較好
		vector_ang[i] = theta;
		//cout << theta << ", ";
	}
	//cout << endl;

	//cout << "angle: " << endl;
	//for (int i = 0; i < vector_ang.size(); i++) {
	//	cout << vector_ang[i] << ", ";
	//}
	//cout << endl;
	//cout << "length: " << endl;
	//for (int i = 0; i < vector_len.size(); i++) {
	//	cout << vector_len[i] << ", ";
	//}
	//cout << endl;

	// ---------- GH Start ----------
	double f = 1;
	vector<int> save_ang_method_2 = method_interval(f, vector_ang, vector_len, false);

	return save_ang_method_2;
	// ---------- GH End ----------
}
vector<int> GH_filter_ours(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt, double& adt_f) {
	// adt_f = default_ransacReprojThreshold_adt
	// adt_f 為直方圖取最大後左右要看的bin數量(根據Inlier rate自適應調整)
	// Vi
	vector<Point2f> vec(original_match_pt.size());
	for (int i = 0; i < original_match_pt.size(); i++) {
		vec[i] = Point2f((target_match_pt[i].x + 1080 - original_match_pt[i].x), (target_match_pt[i].y - original_match_pt[i].y));
	}

	//cout << "[DEBUG] check length:" << endl;
	// length of Vi
	vector<double> vector_len(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double length = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
		vector_len[i] = length;
		//cout << length << ", ";
	}

	// angle of Vi
	vector<double> vector_ang(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double theta = atan2(vec[i].y, vec[i].x);
		//theta = abs(theta * 180.0 / CV_PI);
		theta = theta * 180.0 / CV_PI; // 效果較好
		vector_ang[i] = theta;
		//cout << theta << ", ";
	}
	//cout << endl;

	// ---------- GH Start ----------
	vector<int> save_ang_method_2 = method_interval(adt_f, vector_ang, vector_len, false);
	//vector<int> save_neighbor_method = NN_filter(original_match_pt, target_match_pt, save_ang_method_2);
	return save_ang_method_2;
	// ---------- GH End ----------
}


// only one correspondence in tiny can activate init
vector<double> GH_COOSAC(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt, vector<int> ground_truth, string path, int times) {
	
	//vector<double> constant_list = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	vector<double> constant_list = { 0.2 };

	double reduce_all = 0, time_all = 0, inlier_count_all = 0, inlier_rate_all = 0;
	double recall_all = 0, precision_all = 0, f1_score_all = 0, specificity_all = 0;
	double final_iteration_all = 0, init_iteration_all = 0, tiny_iteration_all = 0, draw_tiny_all = 0, area_fail_all = 0;

	for (int a = 0; a < constant_list.size(); a++) {

		double constant = constant_list[a];

		for (int i = 0; i < times; i++) {

			// ---------- SGH_COOSAC Start ---------- 
			//cout << "---------- SGH_COOSAC ----------" << endl;
			clock_t start, stop;
			start = clock();
			vector<int> output = GH_filter(original, target, original_match_pt, target_match_pt);
			calculate_result(ground_truth, output); // test the result of CC_filter
			cout << "[DEBUG] output size : " << output.size() << endl;

			vector<Point2f> ori(output.size()), tar(output.size());
			cal_method_pt(output, original_match_pt, target_match_pt, ori, tar);
			//draw_img(original, target, ori, tar);
			auto homo_mask = COOSAC(original, target, target_match_pt, original_match_pt, tar, ori, default_confidence, default_maxIters, default_ransacReprojThreshold, constant);
			stop = clock();

			double total_time = ((double)stop - (double)start) / 1000;

			Mat H = homo_mask.H;
			vector<int> mask = homo_mask.mask;
			double final_iteration = homo_mask._final_iteration;
			double init_iteration = homo_mask._init_iteration;
			double tiny_iteration = homo_mask._tiny_iteration;
			double tiny_iteration_sum = homo_mask._tiny_iteration_sum;
			double draw_tiny = homo_mask._draw_tiny;
			double area_fail = homo_mask._area_fail;

			double tiny_HAV_time = homo_mask._tiny_HAV_time;
			double init_HAV_time = homo_mask._init_HAV_time;
			double tiny_random_4_time = homo_mask._tiny_random_4_time;
			double tiny_area_time = homo_mask._tiny_area_time;
			double reduced_random_tiny_time = homo_mask._reduced_random_tiny_time;

			// ---------- SGH_COOSAC End ---------- 

			// ---------- Classification Start ----------
			//cout << "---------- Classification ----------" << endl;
			double inlier_count = accumulate(mask.begin(), mask.end(), 0);

			auto classification = classification_model(ground_truth, mask);
			double TP = classification.TP; double TN = classification.TN;
			double FP = classification.FP; double FN = classification.FN;
			double recall = TP / (TP + FN);
			double precision = TP / (TP + FP);
			double specificity = TN / (TN + FP);
			double f1_score = 2 * precision * recall / (precision + recall);
			// ---------- Classification End ----------
			//cout << "==========================================================================" << endl;

			/* -------Start computing residual of inlier------- */
			//cout << "start computing residual\n";
			vector<double> proj_error_lst(original_match_pt.size());
			computeError(target_match_pt, original_match_pt, H, proj_error_lst);

			double inlier_mres = 0;
			int inlier_count_res = 0;
			double threshold = default_ransacReprojThreshold * default_ransacReprojThreshold;
			for (int i = 0; i < proj_error_lst.size(); i++) {
				//double f = proj_error_lst[i] <= threshold;
				if (proj_error_lst[i] <= threshold) {
					inlier_mres += proj_error_lst[i];
					inlier_count_res += 1;
				}		
			}
			/* -------End of computing residual of inlier------- */

			reduce_all += output.size();
			time_all += total_time;
			inlier_count_all += inlier_count;
			inlier_rate_all += inlier_count / (double)original_match_pt.size();
			recall_all += recall;
			precision_all += precision;
			f1_score_all += f1_score;
			specificity_all += specificity;

			final_iteration_all += final_iteration;
			init_iteration_all += init_iteration;
			tiny_iteration_all += tiny_iteration_sum;
			draw_tiny_all += draw_tiny;
			area_fail_all += area_fail;
		}

		vector<double> accuracy;
		accuracy.push_back(reduce_all / times);
		accuracy.push_back(time_all / times);
		accuracy.push_back(inlier_rate_all / times);
		accuracy.push_back(recall_all / times);
		accuracy.push_back(precision_all / times);
		accuracy.push_back(f1_score_all / times);
		return accuracy;
	}
}
vector<double> GH_COOSAC_ours(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt, vector<int> ground_truth, string path, int times) {
	double default_ransacReprojThreshold_adt;
	vector<double> constant_list = { 0.2 };

	double reduce_all = 0, time_all = 0, inlier_count_all = 0, inlier_rate_all = 0;
	double recall_all = 0, precision_all = 0, f1_score_all = 0, specificity_all = 0;
	double final_iteration_all = 0, init_iteration_all = 0, tiny_iteration_all = 0, draw_tiny_all = 0, area_fail_all = 0;

	for (int a = 0; a < constant_list.size(); a++) {

		double constant = constant_list[a];
		for (int i = 0; i < times; i++) {

			// ---------- SGH_COOSAC Start ---------- 
			//cout << "---------- SGH_COOSAC ----------" << endl;
			clock_t start, stop;
			start = clock();
			vector<int> output = GH_filter_ours(original, target, original_match_pt, target_match_pt, default_ransacReprojThreshold_adt);
			calculate_result(ground_truth, output); // test the result of CC_filter
			
			vector<Point2f> ori(output.size()), tar(output.size());
			cal_method_pt(output, original_match_pt, target_match_pt, ori, tar);
			//draw_img(original, target, ori, tar);
			//cout << default_ransacReprojThreshold_adt << endl;
			auto homo_mask = COOSAC_ours(original, target, target_match_pt, original_match_pt, tar, ori, default_confidence, default_maxIters, default_ransacReprojThreshold_adt, constant);
			
			stop = clock();
			double total_time = ((double)stop - (double)start) / 1000;
			//double total_time = SGH_time + HAV_time;

			Mat H = homo_mask.H;
			vector<int> mask = homo_mask.mask;
			double final_iteration = homo_mask._final_iteration;
			double init_iteration = homo_mask._init_iteration;
			double tiny_iteration = homo_mask._tiny_iteration;
			double tiny_iteration_sum = homo_mask._tiny_iteration_sum;
			double draw_tiny = homo_mask._draw_tiny;
			double area_fail = homo_mask._area_fail;

			double tiny_HAV_time = homo_mask._tiny_HAV_time;
			double init_HAV_time = homo_mask._init_HAV_time;
			double tiny_random_4_time = homo_mask._tiny_random_4_time;
			double tiny_area_time = homo_mask._tiny_area_time;
			double reduced_random_tiny_time = homo_mask._reduced_random_tiny_time;

			// ---------- SGH_COOSAC End ---------- 

			// ---------- Classification Start ----------
			double inlier_count = accumulate(mask.begin(), mask.end(), 0);
			auto classification = classification_model(ground_truth, mask);
			double TP = classification.TP; double TN = classification.TN;
			double FP = classification.FP; double FN = classification.FN;
			double recall = TP / (TP + FN);
			double precision = TP / (TP + FP);
			double specificity = TN / (TN + FP);
			double f1_score = 2 * precision * recall / (precision + recall);
			cout << "[Final Result]\n";
			cout << "F1-score is " << f1_score << " and Specificity is " << specificity << endl;
			cout << "Recall is " << recall << " and Precision is " << precision << endl << endl;
			cout << "H: \n" << H << endl;
			// ---------- Classification End ----------
			//cout << "==========================================================================" << endl;

			/* Saving output result */
			//cout << "start drawing\n";
			//draw_output(original, target, original_match_pt, target_match_pt, ground_truth, mask);
			//cout << "end drawing\n";
			/*----------------------*/

			/* ------- Start drawing result correspondence ------- */
			//cout << "start draw result\n";
			//draw_result(original, target, original_match_pt, target_match_pt, mask);
			//cout << "end draw result\n";
			/* ------- End of drawing result correspondence ------- */

			/* -------Start computing residual of inlier------- */
			//cout << "start computing residual\n";
			vector<double> proj_error_lst(original_match_pt.size());
			computeError(target_match_pt, original_match_pt, H, proj_error_lst);

			double inlier_mres = 0;
			int inlier_count_res = 0;
			double threshold = default_ransacReprojThreshold * default_ransacReprojThreshold;
			for (int i = 0; i < proj_error_lst.size(); i++) {
				//double f = proj_error_lst[i] <= threshold;
				if (proj_error_lst[i] <= threshold) {
					inlier_mres += proj_error_lst[i];
					inlier_count_res += 1;
				}
			}
			cout << "inlier number : " << inlier_count_res << endl;
			cout << "Mean residual of inlier : " << inlier_mres / inlier_count_res << endl;

			/* -------End of computing residual of inlier------- */

			reduce_all += output.size();
			time_all += total_time;
			inlier_count_all += inlier_count;
			inlier_rate_all += inlier_count / (double)original_match_pt.size();
			recall_all += recall;
			precision_all += precision;
			f1_score_all += f1_score;
			specificity_all += specificity;

			final_iteration_all += final_iteration;
			init_iteration_all += init_iteration;
			tiny_iteration_all += tiny_iteration_sum;
			draw_tiny_all += draw_tiny;
			area_fail_all += area_fail;
		}

		//ofs << "\n\n" << endl;

		vector<double> accuracy;
		accuracy.push_back(reduce_all / times);
		accuracy.push_back(time_all / times);
		accuracy.push_back(inlier_rate_all / times);
		accuracy.push_back(recall_all / times);
		accuracy.push_back(precision_all / times);
		accuracy.push_back(f1_score_all / times);
		return accuracy;
	}
}