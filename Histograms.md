# Histogram

## 1 1D Histogram

```C++
MatND histogram;
int histSize = 256;
const int* channel_numbers = { 0 };
float channel_range[] = { 0.0, 256.0 };
const float* channel_ranges = channel_range;
int number_bins = histSize;

calcHist(&image3, 1, 0, Mat(), histogram, 1, &number_bins, &channel_ranges);
```

Equalisation, make your picture brighter.

```C++
Mat histEqualized;
equalizeHist(img, histEqualized);

imshow();
waitKey();
```

comparision for two picture: like a classification.

```C++
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

int main() {
	string path1 = "../data/airplane.png";
	string path2 = "../data/dog.png";
	string path3 = "../data/coins2.jpeg";
	string path4 = "../data/water_coins.jpg";
	string path5 = "../data/coins.jpeg";

	Mat img1 = imread(path1, IMREAD_COLOR);
	Mat img2 = imread(path2, IMREAD_COLOR);
	Mat img3 = imread(path3, IMREAD_GRAYSCALE);
	Mat img4 = imread(path4, IMREAD_GRAYSCALE);
	Mat img5 = imread(path5, IMREAD_GRAYSCALE);

	MatND histogram;
	int histSize = 256;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 256.0 };
	const float* channel_ranges = channel_range;
	int number_bins = histSize;

	calcHist(&img3, 1, 0, Mat(), histogram, 1, &number_bins, &channel_ranges);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	MatND histogram2;

	calcHist(&img5, 1, 0, Mat(), histogram2, 1, &number_bins, &channel_ranges);
	Mat histImage2(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(histogram2, histogram2, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++) {
		line(histImage2, Point(bin_w * (i - 1), hist_h - cvRound(histogram2.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram2.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	double histMatchingCorrelation = compareHist(histogram, histogram2, HISTCMP_CORREL);
	double histMatchingChiSquare = compareHist(histogram, histogram2, HISTCMP_CHISQR);
	double histMatchingIntersect = compareHist(histogram, histogram2, HISTCMP_INTERSECT);
	double histMatchingBhattacharyya = compareHist(histogram, histogram2, HISTCMP_BHATTACHARYYA);

	cout << "correlation: " << histMatchingCorrelation << endl;
	cout << "chisquare: " << histMatchingChiSquare << endl;
	cout << "intersect: " << histMatchingIntersect << endl;
	cout << "bhattacharyya: " << histMatchingBhattacharyya << endl;

	//imshow("source image", img3);
	//imshow("calcHist 1", histImage);
	//imshow("source image 2", img5);
	//imshow("calcHist 2", histImage2);

	// equalize the histogram
	Mat histEqualized;
	equalizeHist(img4, histEqualized);
	//imshow("img4", img4);
	//imshow("equalized img", histEqualized);

	vector<Mat> bgr_planes;
	split(img1, bgr_planes);
	float range[] = { 0, 256 }; // the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat histImage3(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage3.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage3.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage3.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++) {
		line(histImage3, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage3, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage3, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("img1", img1);
	imshow("histImage", histImage3);
	waitKey(0);
	return 0;
}
```

![](D:\SZTU\大三\ComputerVisionCourse-main\note\pic\histogram.png)

![](D:\SZTU\大三\ComputerVisionCourse-main\note\pic\equalized.png)

haven't watch video 7!!