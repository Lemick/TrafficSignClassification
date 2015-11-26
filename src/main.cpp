/// main.cpp : Defines the entry point for the console application.
///
///-------------------------------In the name of GOD
/// Author: Mickael Beguin, from a source code of Mohammad Reza Mostajabi
///-------------------------------
/// BOW+SVM : this program provide an easy way to create a "Vocabulary" of features extracted with SURF detector
/// from a set of images and then to create a Support Vector Machine using the Bag Of Words methods
///-------------------------------
/// How to use: Set the "define"

#define NB_IMAGES 20
#define NB_CLASSES 3


#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;


const std::string trainPath = "/trainsign/";
const std::string evalPath = "/trainsign/";

//HEADER
bool writeVocabulary( const string& filename, const Mat& vocabulary );


char ch[150];
//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionarr
Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
SurfFeatureDetector detector(500);

//---dictionary size=number of cluster's centroids
int dictionarySize = 1500;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);



void collectclasscentroids() {
    Mat img;
    int i,j;
    for(j=1;j<= NB_CLASSES ;j++)
        for(i=1;i<= NB_IMAGES ;i++){
            sprintf( ch,"%s%d%s%d%s", trainPath.c_str(),j," (",i,").jpg");
            const char* imageName = ch;
            img = imread(imageName,0);

            cout << "rows " << img.rows << "cols " << img.cols << endl;

            vector<KeyPoint> keypoint;
            detector.detect(img, keypoint);
            Mat features;
            extractor->compute(img, keypoint, features);
            bowTrainer.add(features);
        }
    return;
}



int _tmain(int argc, _TCHAR* argv[])
{

    int i,j;
    Mat img2;
    cout<<"Vector quantization..."<<endl;
    collectclasscentroids();
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    int count=0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    cout<<"Clustering "<<count<<" features"<<endl;
    //choosing cluster's centroids as dictionary's words
    Mat dictionary = bowTrainer.cluster();
    //save da vocabulary
    writeVocabulary( "vocabulary", dictionary );

    bowDE.setVocabulary(dictionary);
    cout<<"extracting histograms in the form of BOW for each image "<<endl;
    Mat labels(0, 1, CV_32FC1);
    Mat trainingData(0, dictionarySize, CV_32FC1);
    int k=0;
    vector<KeyPoint> keypoint1;
    Mat bowDescriptor1;
    //extracting histogram in the form of bow for each image
    for(j=1;j<= NB_CLASSES ;j++)
        for(i=1;i<= NB_IMAGES ;i++){


            sprintf( ch,"%s%d%s%d%s",trainPath.c_str(),j," (",i,").jpg");
            const char* imageName = ch;
            img2 = imread(imageName,0);

            detector.detect(img2, keypoint1);
            bowDE.compute(img2, keypoint1, bowDescriptor1);

            cout << "rows " << bowDescriptor1.rows << "cols " << bowDescriptor1.cols << endl;

            trainingData.push_back(bowDescriptor1);

            labels.push_back((float) j);
        }
    cout << "rows training dat" << trainingData.rows << "cols" << trainingData.cols << endl;


    //Setting up SVM parameters
    CvSVMParams params;
    params.kernel_type=CvSVM::RBF;
    params.svm_type=CvSVM::C_SVC;
    params.gamma=0.50625000000000009;
    params.C=312.50000000000000;
    params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
    CvSVM svm;



    printf("%s\n","Training SVM classifier");

    bool res=svm.train(trainingData,labels,cv::Mat(),cv::Mat(),params); //DEPRECIATED: use train_auto
    svm.save("Bow_svm_alpha");
    cout<<"Processing evaluation data..."<<endl;


    Mat groundTruth(0, 1, CV_32FC1);
    Mat evalData(0, dictionarySize, CV_32FC1);
    k=0;
    vector<KeyPoint> keypoint2;
    Mat bowDescriptor2;


    Mat results(0, 1, CV_32FC1);;
    for(j=1;j<= NB_CLASSES ;j++)
        for(i=1;i<=NB_IMAGES;i++){


            sprintf( ch,"%s%d%s%d%s",trainPath.c_str(),j," (",i,").jpg");
            const char* imageName = ch;
            img2 = imread(imageName,0);

            detector.detect(img2, keypoint2);
            bowDE.compute(img2, keypoint2, bowDescriptor2);

            evalData.push_back(bowDescriptor2);
            groundTruth.push_back((float) j);
            float response = svm.predict(bowDescriptor2);
            cout << "respones :  " << response << endl;
            results.push_back(response);
        }



    //calculate the number of unmatched classes
    double errorRate = (double) countNonZero(groundTruth- results) / evalData.rows;
    printf("%s%f","Error rate is ",errorRate);
    return 0;

}

bool writeVocabulary( const string& filename, const Mat& vocabulary )
{
    cout << "Saving vocabulary..." << endl;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}
