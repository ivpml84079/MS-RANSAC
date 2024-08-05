# MS-RANSAC
Fast and Accurate Early Termination RANSAC for Image Feature Matching Using Machine Learning-Based Reliable Correspondence Set by Kuo-Liang Chung, Chia-Chi Hsu, and Yu-Chi Chang.

A C++ implementation of the GH-COOSAC method for removing mismatches in image feature matching.

![img](github_image.jpg)

# Dependencies
* OpenCV 4.8.0

# Testing enviroment
* Windows 10
* Visual Studio 2022
* C++ 14

# Support Vector Machine (SVM)
```svm_train_main.cpp``` is used to train an SVM classifier. By default, the trained model will be saved in ```/GH-COOSAC-main/svm_model```.
You can modify the code to change the save directory:
```
svm_save_model(your_path, model);
```

# Contact
If you have any questions, please email us via

Chia-Chi Hsu: <m11115040@mail.ntust.edu.tw>

Kuo-Liang Chung: <klchung01@gmail.com>

