#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <complex.h>
using namespace cv;
using namespace std;
class Person;
class Face;

void detectAndCut(Mat image, int batch, int num);
void readFaces();
void giveEigen();
void faceRecognize();
void readTestImages();

string xmlPath = "haarcascade_frontalface_alt.xml";	// using for face detection
string forTesting;
vector<Person*> personInAll;
vector<Face*> faceInAll;
vector<Mat> faceMats;
Size standard = Size(25, 25);
Mat meanMat = Mat();
Mat covMat = Mat();
Mat eValueMat = Mat();
Mat eVectorMat = Mat();
vector<Mat> eFaces;
Mat m0;	// mapping space
Mat m0T;	// mapping
float energyPercent; //Energy percentage
int eCount;
Mat testImage;

class Person
{
public:
	vector<Face *> faces;
	int num;
	explicit Person(int num)
	{
		this->num = num;
	}
};

class Face
{
public:
	int num;
	Mat image;
	Person *person;
	Mat coord;
	explicit Face(Person *p, int num)
	{
		this->person = p;
		this->num = num;
	}
};

int main(int argc, char**argv)
{
	//CascadeClassifier a;
	//if (!a.load(xmlPath))
	//{
	//	cout << "Can't read xml file!" << endl;
	//	return 1;
	//}

	//for (int j = 1; j <= 10; j++)
	//{
	//	for (int i = 1; i <= 11; i++)
	//	{
	//		string name = "./FaceLib/" + to_string(j) + "/" + to_string(i) + ".tiff";
	//		Mat image = imread(name);
	//		detectAndCut(image, j, i); // face detection
	//	}
	//}

	energyPercent = 0.03;
	forTesting = "./detected/3/11.tiff";
	readFaces();
	readTestImages();
	giveEigen();
	faceRecognize();
	waitKey(0);
	return 0;
}

void detectAndCut(Mat image, int batch, int num)
{
	CascadeClassifier ccf;
	ccf.load(xmlPath);
	vector<Rect> faces;
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);
	ccf.detectMultiScale(gray, faces, 1.1, 3, 0, Size(50, 50), Size(500, 500));
	Mat image1, image2;

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		image1 = image(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
	}
	resize(image1, image2, Size(100, 100));
	imwrite("./detected/" + to_string(batch) + "/" + to_string(num) + ".tiff", image2);
	cvWaitKey(0);
	cout << "PERSON: " << batch << " FACE: " << num << " Processing Complete." << endl;
}

void readFaces()
{
	cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	cout << "Begin Reading Faces..." << endl;
	for(int i = 1; i <= 10; i++)
	{
		Person *pt = new Person(i);
		personInAll.push_back(pt);
		for(int j = 1; j <= 10; j++)
		{
			Face *ft = new Face(pt, j);
			Mat image = imread("./detected/" + to_string(i) + "/" + to_string(j) + ".tiff", CV_LOAD_IMAGE_GRAYSCALE);
			resize(image, image, standard);
			ft->image = Mat(standard.height, standard.width, CV_8UC1);
			image.copyTo(ft->image);
			normalize(ft->image, ft->image, 255, 0, NORM_MINMAX);
			faceMats.push_back(ft->image);
			pt->faces.push_back(ft);
			faceInAll.push_back(ft);
			cout << "PERSON: " << i << " FACE: " << j << " Reading Complete." << endl;
		}
	}
	cout << "Reading Faces Complete." << endl;
	cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
}

void readTestImages()
{
	cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	cout << "Reading Testing Faces..." << endl;
	testImage = imread(forTesting, CV_LOAD_IMAGE_GRAYSCALE);
	resize(testImage, testImage, standard);
	normalize(testImage, testImage, 255, 0, NORM_MINMAX);
	cout << "Reading Faces Complete." << endl;
	cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
}

void giveEigen()
{
	cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	cout << "Calculating Eigen Faces..." << endl;
	calcCovarMatrix(faceMats, covMat, meanMat, COVAR_NORMAL); // Calculate the covariance matrix
	Mat mImage;
	meanMat.convertTo(mImage, CV_8UC1);
	imshow("Average", mImage);
	imwrite("./result/average/average.jpg", mImage);
	eigen(covMat, eValueMat, eVectorMat); // Calculate the eigenmatrix
	eCount = eVectorMat.rows * energyPercent; // Figure out how much percentage eigenfaces you end up with
	cout << "Number of Eigen: " << eCount << endl;
	m0T = Mat(eCount, standard.height * standard.width, CV_64F);
	m0 = Mat(standard.height * standard.width, eCount, CV_64F);
	for (int i = 0; i < eCount; i++)
	{
		Mat tmat0 = Mat(standard.height, standard.width, CV_64F);
		Mat tmat1 = Mat(standard.height, standard.width, CV_8UC1); // make it 8U for display
		for(int j = 0; j < standard.width * standard.height; j++)
		{
			tmat0.at<double>(j / standard.width, j % standard.width) = eVectorMat.at<double>(i, j);
			m0T.at<double>(i ,j) = eVectorMat.at<double>(i, j);
			m0.at<double>(j, i) = eVectorMat.at<double>(i, j);
		}
		normalize(tmat0, tmat0, 255, 0, NORM_MINMAX);
		tmat0.convertTo(tmat1, CV_8UC1);
		eFaces.push_back(tmat1);
	}
	for (int i = 0; i < 10 && i < eCount; i++) {
		imshow("EigenFace" + to_string(i), eFaces.at(i));
		imwrite("./result/eigen/" + to_string(i) + ".jpg", eFaces.at(i));
	}
	cout << "Calculating Eigens Complete." << endl;
	cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
}

void faceRecognize()
{
	cout << testImage.reshape(0, 1).rows << "*" << testImage.reshape(0, 1).cols << endl;
	cout << m0.rows << "*" << m0.cols << endl;
	Mat testDMat;
	testImage.reshape(0, 1).convertTo(testDMat, CV_64F);
	Mat testCod = testDMat * m0; // calculate the coordinates
	double minimumDist = -1;
	Face *recogResult = nullptr;
	cout << testCod.rows << "*" << testCod.cols << endl;
	for(vector<Face *>::iterator it= faceInAll.begin(); it != faceInAll.end(); it++)
	{
		double tdist;
		Face *tface = *it;
		Mat tMat;
		tface->image.reshape(0, 1).convertTo(tMat, CV_64F);
		tface->coord = tMat * m0;
		tdist = 0;
		for (int i = 0; i < eCount; i++)
		{
			tdist += pow(tface->coord.at<double>(0, i) - testCod.at<double>(0, i), 2);
		}
		cout << tface->person->num << ": " << tdist << endl;
		if (tdist < minimumDist || minimumDist == -1) {
			minimumDist = tdist;
			recogResult = tface;
		}
	}
	cout << "result:\n" << "person" << recogResult->person->num << " img" << recogResult->num << endl;
	imshow("most similar face", recogResult->image);
	Mat colorfulTestImg;
	cvtColor(testImage, colorfulTestImg, COLOR_GRAY2BGR);
	putText(colorfulTestImg, to_string(recogResult->person->num), cvPoint(2, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	imshow("test image", colorfulTestImg);
}