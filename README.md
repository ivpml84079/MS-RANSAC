# MS-RANSAC
Fast and Accurate Early Termination RANSAC for Image Feature Matching Using Machine Learning-Based Reliable Correspondence Set by Kuo-Liang Chung, Chia-Chi Hsu, and Yu-Chi Chang


![img](github_image.jpg)

# Dependencies
* OpenCV 4.8.0

# Testing enviroment
* Windows 10
* Visual Studio 2022
* C++ 14


# Usage
After creating the project with Visual Studio 2022, compile the codes of our method in the project directory, and run ```Source.cpp```.


# Support Vector Machine (SVM)
The two training sets, ```train_orientation_0729.txt``` and ```train_length_0729.txt```, are used to train the SVM-based orientation model ```train_orientation_0729.model``` and the SVM-based length model ```train_length_0729.model```, respectively. Both trained SVM models are contained in the MLGH-based classifier ```svm_train_main.cpp```.

The trained MLGH-based classifier is saved in ```/MS-RANSAC-main/svm_model```.

# Contact
If you have any questions, please email us via

Yu-Chi Chang: <gigi1060307@gmail.com>

Kuo-Liang Chung: <klchung01@gmail.com>

