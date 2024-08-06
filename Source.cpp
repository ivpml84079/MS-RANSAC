#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <numeric>
#include <cstring>

#include "method.h"
#include "ransac.h"
#include "draw.h"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

/* Process 2/12
	1. �w�Q��inlier rate�վ�bin���k�ѦҼƥءBRANSAC inlier threshold
*/
/* Problem 2/12
	1. �b�Cinlier rate��precision���C�A�i��Oransac threshold�դӤj�ɭP(�ثe�b0.1�ɴN�|+1)
*/
void load_UAV_file(string root, vector<string>& src_data, vector<string>& tar_data) {
	// load delete list
	set<string> delete_list;
	ifstream f_delete(root + "delete_list.txt");
	string line;
	while (getline(f_delete, line)) 
		delete_list.insert(line);
	//delete_list.insert("IMG_0129.JPG");
	//delete_list.insert("IMG_0123.JPG");


	//delete_list.clear();
	// load data number except delete data
	for (int i = 0; i < 200; i++) {
		char src_idx[5];
		char tar_idx[5];
		sprintf_s(src_idx, "%04d", i);
		strcpy_s(tar_idx, sizeof(tar_idx), src_idx);
		tar_idx[0] = '1';
		ifstream f(root + "IMG_" + string(src_idx) + ".JPG");
		if (f.good() && delete_list.find("IMG_" + string(src_idx) + ".JPG")==delete_list.end()) {
			src_data.push_back("IMG_" + string(src_idx) + ".JPG");
			tar_data.push_back("IMG_" + string(tar_idx) + ".JPG");
		}
	}

	return;
}

int main(int argc, char* argv[]) {
	// 0 : YC, 1 : Ours
	int mode = 1;
	////////////////////
	clock_t start, stop;
	vector<string> dataset_1, dataset_2;
	//vector<float> adjust_inlier_rate = { 0.2 };
	vector<float> adjust_inlier_rate = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	double times = 1;  
	set<string> delete_list;

	// "Airport" "Small_Village" "University_Campus" "UAV"
	string dataset = "UAV";

	if (dataset == "Airport") {
		dataset_1 = { "IMG_0061.JPG","IMG_0116.JPG","IMG_0177.JPG","IMG_0282.JPG","IMG_3479.JPG" };
		dataset_2 = { "IMG_0062.JPG","IMG_0117.JPG","IMG_0178.JPG","IMG_0283.JPG","IMG_3480.JPG" };
	}
	else if (dataset == "Small_Village") {
		dataset_1 = { "IMG_0924.JPG","IMG_0970.JPG","IMG_1011.JPG","IMG_1113.JPG","IMG_1204.JPG" };
		dataset_2 = { "IMG_0925.JPG","IMG_0971.JPG","IMG_1012.JPG","IMG_1114.JPG","IMG_1205.JPG" };
	}
	else if (dataset == "University_Campus") {
		dataset_1 = { "IMG_0060.JPG","IMG_0098.JPG","IMG_0172.JPG","IMG_0333.JPG","IMG_0403.JPG" };
		dataset_2 = { "IMG_0061.JPG","IMG_0099.JPG","IMG_0173.JPG","IMG_0334.JPG","IMG_0404.JPG" };
	}

	else if(dataset == "UAV") {
		//dataset_1 = { "IMG_0062.JPG"};
		//dataset_2 = { "IMG_1062.JPG"};
		cout << "collecting UAV dataset...\n";
		load_UAV_file("C:\\CC_temp\\UAV\\", dataset_1, dataset_2);
		cout << "UAV dataset size : " << dataset_1.size() << endl;
		//for (int i = 0; i < dataset_1.size(); i++)
		//	cout << dataset_1[i] << " -> " << dataset_2[i] << endl;
		//return 0;
	}
	else{
		cout << "You need to choose a dataset." << endl;
	}

	
	cout << "openCV Ver. : " << CV_VERSION << endl;

	ofstream ofs;
	if(mode == 0)
		ofs.open("C:\\CC_temp\\COOSAC\\" + dataset + "\\total_1.csv", fstream::out);
	else if (mode ==1)
		ofs.open("C:\\CC_temp\\COOSAC\\" + dataset + "\\total_ours_fianl_tmp.csv", fstream::out);
	//ofs.open(".\\COOSAC\\" + dataset + "\\total.csv", fstream::out);

	ofs << "GT inlier" << "," << "Correspondence" << "," << "Adjust" << "," << "Reduce" << ","
		<< "Total time" << ", " << "Inlier rate" << ","
		<< "Recall" << "," << "Precision" << "," << "F1-score" << endl;


	for (int adjustIdx = 0; adjustIdx < adjust_inlier_rate.size(); adjustIdx++) {
		double GT_inlier_all = 0, correspondence_all = 0, adjust_all = 0, reduce_all = 0,
			time_all = 0, inlier_rate_all = 0, recall_all = 0, precision_all = 0, f1_score_all = 0;

		for (int datasetIdx = 0; datasetIdx < dataset_1.size(); datasetIdx++) {
			Mat original = imread("..\\" + dataset + "\\" + dataset_1[datasetIdx]);
			Mat target = imread("..\\" + dataset + "\\" + dataset_2[datasetIdx]);
			//resize(original, original, Size(1080, 720), INTER_LINEAR);
			//resize(target, target, Size(1080, 720), INTER_LINEAR);
			//draw(original, target);
			cout << "=============================================\n";
			cout << "[DEBUG] data number : " << dataset_1[datasetIdx] << endl;

			start = clock();
			vector<Point2f> all_original_match_pt, all_target_match_pt;
			vector<int> all_ground_truth;
			vector<int> remove_list;
			fstream file;
			string line;

			file.open("..\\" + dataset + "\\" + dataset_1[datasetIdx].substr(4, 4) + "_pts.csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				vector<string> tmp;
				while (getline(templine, data, ','))
					tmp.push_back(data);
				all_original_match_pt.push_back(Point2f((stof(tmp[0])), (stof(tmp[1]))));
			}
			file.close();

			file.open("..\\" + dataset + "\\" + dataset_2[datasetIdx].substr(4, 4) + "_pts.csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				vector<string> tmp;
				while (getline(templine, data, ','))
					tmp.push_back(data);
				all_target_match_pt.push_back(Point2f((stof(tmp[0])), (stof(tmp[1]))));
			}
			file.close();

			file.open("..\\" + dataset + "\\" + dataset_1[datasetIdx].substr(4, 4) + "_" + dataset_2[datasetIdx].substr(4, 4) + ".csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				while (getline(templine, data, ',')) {
					if (data == "False")
						all_ground_truth.push_back(0);
					else
						all_ground_truth.push_back(1);
				}
			}
			file.close();
			//cout << "all_ground_truth = " << all_ground_truth.size() << endl;

			stringstream stream;
			stream.precision(1);
			stream << fixed;
			stream << adjust_inlier_rate[adjustIdx];
			string str_adjust = stream.str();
			file.open("..\\" + dataset + "\\adjust\\" + dataset_1[datasetIdx].substr(4, 4) + "_" + dataset_2[datasetIdx].substr(4, 4) + "_" + str_adjust + ".csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				vector<string> tmp;
				while (getline(templine, data, ','))
					tmp.push_back(data);
				remove_list.push_back(stoi(tmp[0]));
			}
			file.close();

			vector<Point2f> original_match_pt(all_original_match_pt), target_match_pt(all_target_match_pt);
			vector<int> ground_truth(all_ground_truth);

			for (int remove_index = remove_list.size() - 1; remove_index >= 0; remove_index--) {
				original_match_pt.erase(original_match_pt.begin() + remove_list[remove_index]);
				target_match_pt.erase(target_match_pt.begin() + remove_list[remove_index]);
				ground_truth.erase(ground_truth.begin() + remove_list[remove_index]);
			}
			double gt_inlier = accumulate(ground_truth.begin(), ground_truth.end(), 0);
			stop = clock();
			//cout << "Time is " << ((double)stop - (double)start) * 0.001 << endl;

			cout << "GT inlier rate : " << gt_inlier / ground_truth.size() << endl;

			/*cout << dataset_1[datasetIdx].substr(4, 4) << "_" << dataset_2[datasetIdx].substr(4, 4) <<
				": Inlier rate in ground truth = " << adjust_inlier_rate[adjustIdx] << ", size = " << ground_truth.size() << endl;*/

			string path = "..\\COOSAC\\" + dataset + "\\" + dataset_1[datasetIdx].substr(4, 4) + "_" + dataset_2[datasetIdx].substr(4, 4);

			//draw_img(original, target, original_match_pt, target_match_pt);
			vector<double> accuracy;
			if (mode == 0)
				accuracy = GH_COOSAC(original, target, original_match_pt, target_match_pt, ground_truth, path, times);
			else if(mode == 1)
				accuracy = GH_COOSAC_ours(original, target, original_match_pt, target_match_pt, ground_truth, path, times);
				
			GT_inlier_all += gt_inlier;
			correspondence_all += ground_truth.size();
			adjust_all += adjust_inlier_rate[adjustIdx];
			reduce_all += accuracy[0];
			time_all += accuracy[1];
			inlier_rate_all += accuracy[2];
			recall_all += accuracy[3];
			precision_all += accuracy[4];
			f1_score_all += accuracy[5];


		}
		cout << "inlier rate in this round : " << adjust_inlier_rate[adjustIdx] << endl;
		cout << "GT inlier = " << GT_inlier_all / dataset_1.size() << ", Correspondence = " << correspondence_all / dataset_1.size()
			<< ", Adjust = " << adjust_all / dataset_1.size() << ", Reduce = " << reduce_all / dataset_1.size() << endl
			<< ", Total time = " << time_all / dataset_1.size() << ", Inlier rate = " << inlier_rate_all / dataset_1.size()
			<< ", Recall = " << recall_all / dataset_1.size() << ", Precision = " << precision_all / dataset_1.size() 
			<< ", F1-score = " << f1_score_all / dataset_1.size() << endl;

		ofs << GT_inlier_all / dataset_1.size() << "," << correspondence_all / dataset_1.size() << "," << adjust_all / dataset_1.size() << ","
			<< reduce_all / dataset_1.size() << "," << time_all / dataset_1.size() << "," << inlier_rate_all / dataset_1.size() << ","
			<< recall_all / dataset_1.size() << "," << precision_all / dataset_1.size() << "," << f1_score_all / dataset_1.size() << endl;

		cout << "======================================================" << endl;
	}
	
	return 0;
}
