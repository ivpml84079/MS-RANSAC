#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "svm.h"

using namespace std;

vector<string> split(string& str, string& pattern) {
    vector<string> result;
    string::size_type begin, end;

    end = str.find(pattern);
    begin = 0;

    while (end != string::npos) {
        if (end - begin != 0) {
            result.push_back(str.substr(begin, end - begin));
        }
        begin = end + pattern.size();
        end = str.find(pattern, begin);
    }

    if (begin != str.length()) {
        result.push_back(str.substr(begin));
    }
    return result;
}

void read_data(string filename, svm_problem& prob, vector<svm_node>& x_space) {
    ifstream file(filename);
    string line;
    vector<int> labels;
    string pattern = " ";

    int idx = 0;
    while (getline(file, line)) {
        vector<string> ret = split(line, pattern);
        labels.push_back(stoi(ret[0]));

        for (int i = 1; i <= 8; i++) {
            if ( i == 8 ) x_space[idx].index = -1;
            else {
                int end = ret[i].size()-2;
                string value = ret[i].substr(2, end);
                x_space[idx].index = i;
                x_space[idx].value = stoi(value);
            } // else
            idx++;
        } // for
    } // while

    prob.l = labels.size();
    prob.y = new double[prob.l];
    prob.x = new svm_node * [prob.l];

    for (int i = 0; i < prob.l; i++) {
        prob.y[i] = labels[i];
        prob.x[i] = &x_space[i*8];
    } // for
}

vector<vector<svm_node>> read_test_data(string& filename) {
    vector<vector<svm_node>> test_samples;
    ifstream file(filename);
    string line;
    vector<int> labels;
    string pattern = ",";

    int idx = 0;
    while (getline(file, line)) {
        vector<string> ret = split(line, pattern);
        vector<svm_node> tmp(8);
        
        tmp[0].index = 1;
        tmp[0].value = stoi(ret[0]);
        tmp[1].index = 2;
        tmp[1].value = stoi(ret[1]);
        tmp[2].index = 3;
        tmp[2].value = stoi(ret[2]);
        tmp[3].index = 4;
        tmp[3].value = stoi(ret[3]);
        tmp[4].index = 5;
        tmp[4].value = stoi(ret[4]);
        tmp[5].index = 6;
        tmp[5].value = stoi(ret[5]);
        tmp[6].index = 7;
        tmp[6].value = stoi(ret[6]);
        tmp[7].index = -1;
        tmp[7].value = 0;

        test_samples.push_back(tmp);
    } // while
    
    return test_samples;
}


int main() {
    
    // 读取数据
    svm_problem prob;
    int size = 2268*8;
    vector<svm_node> x_space(size);
    
    //string str = "C:\\GH-COOSAC-main\\svm_training_txt\\train_orientation_0729.txt";
    string str = "C:\\GH-COOSAC-main\\svm_training_txt\\train_length_0729.txt";
    read_data(str, prob, x_space);
    
    // 設置 SVM 參數
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.gamma = 0.5;
    param.C = 1;
    param.cache_size = 100;
    param.eps = 1e-3;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.shrinking = 1;
    param.probability = 0;

    // 檢查參數是否正確
    const char* error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
        cerr << "error: " << error_msg << std::endl;
        return 1;
    }
    
    // 訓練模型
    svm_model* model = svm_train(&prob, &param);
    
    if (svm_save_model("C:\\GH-COOSAC-main\\svm_model\\train_length_0729.model", model)!=0) {
        cerr << "Error: unable to save model to file." << endl;
    } // if
   


    // 釋放内存
    svm_free_and_destroy_model(&model);
    delete[] prob.y;
    delete[] prob.x;

    return 0;
}
