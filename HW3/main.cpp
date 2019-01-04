#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
class Person;
class Face;

void detectAndDisplay(Mat image, int batch, int num);
void readFaces();
void giveEigen();
void faceRecognize();

string xmlPath = "haarcascade_frontalface_alt.xml";	// using for face detection
string imagePath;
vector<Person*> personInAll;
vector<Face*> faceInAll;
vector<Mat> faceMats;
Size standardSize = Size(25, 25);
Mat meanMat = Mat();
Mat covMat = Mat();
Mat eValueMat = Mat();
Mat eVectorMat = Mat();
vector<Mat> eFaces;
Mat AT;	// mapping space
Mat A;	// mapping
float energyPercent;
int eCount;

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
	CascadeClassifier a;
	if (!a.load(xmlPath))
	{
		cout << "Can't read xml file!" << endl;
		return 1;
	}

	for(int j = 1; j <= 10; j++)
	{
		for (int i = 1; i <= 10; i++)
		{
			string name = "./FaceLib/" + to_string(j) + "/" + to_string(i) + ".tiff";
			Mat image = imread(name);
			detectAndDisplay(image, j, i); // face detection
		}
	}

	energyPercent = atof(argv[1]);
	imagePath = String(argv[2]);
	readFaces();
	giveEigen();
	faceRecognize();
	waitKey(0);
	return 0;
	return 0;
}

void detectAndDisplay(Mat image, int batch, int num)
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
	cout << "BATCH: " << batch << " FACE: " << num <<" Processing Complete." << endl;
}