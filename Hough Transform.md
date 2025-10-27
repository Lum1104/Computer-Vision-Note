---
layout: default
title: Hough Transform
nav_order: 10
permalink: /hough-transform/
description: Line and circle detection using the Hough transform in OpenCV.
---

- Hough Transform Lines
- Hough Transform Circles
- Hough Generalized
- Direct transformation from image space to probability of the existence of some feature

## Line Detection

Equations that represents lines

- Lines in Hough space
- Sinusoidal curves
- Cartesian or polar coordinates

## Circle Detection

Roughly same approach as with lines

- A line defined by 2 parameters
- For a circle we need 3 parameters
- OpenCV uses Hough Gradient Method
  - more efficient
  - Divided in two stages
    - Edge Detection and finding possible circle centers
    - Finding the best radius for each candidate center

 $C: (x_{\text{center}}, y_{\text{center}}, r)$ 

```cpp
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>

using namespace std;
using namespace cv;

int main() {
	Mat imgLines, imgCircles, detectImgLines, detectImgLinesP, detectImgCircles;
	imgLines = imread("../data/cameraman.tif");
	imgCircles = imread("../data/coins.jpeg");

	imgLines.copyTo(detectImgLines);
	imgLines.copyTo(detectImgLinesP);
	imgCircles.copyTo(detectImgCircles);

	// Line Detection Hough Transform
	// Edge detection
	Canny(imgLines, detectImgLines, 200, 255);
	// imshow("canny", detectImgLines);

	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(detectImgLines, lines, 1, CV_PI, 150); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(detectImgLines, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}

	//imshow("cameraman", imgLines);
	//imshow("lines", detectImgLines);

	// Probabilistic Line Transform
	Canny(imgLines, detectImgLinesP, 200, 255);
	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(detectImgLinesP, linesP, 1, CV_PI / 180, 50, 50, 10);

	for (size_t i = 0; i < linesP.size(); i++) {
		Vec4i l = linesP[i];
		line(detectImgLinesP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
	}

	//imshow("P", detectImgLinesP);

	// Circle Detection Hough Transform 
	Mat gray;
	cvtColor(imgCircles, gray, COLOR_BGR2GRAY);
	medianBlur(gray, gray, 5);

	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
		gray.rows / 4, // change this value to detect circles with different distances to each other
		100, 30, 100, 150 // change the last two parameters (min_radius & max_radius) to detect larger circles
	);
	for (size_t i = 0; i < circles.size(); i++) {
		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		// circle center
		circle(detectImgCircles, center, 1, Scalar(0, 0, 255), 3, LINE_AA);
		// circle outline
		int radius = c[2];

		circle(detectImgCircles, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
	}
	imshow("circle", detectImgCircles);
	imshow("original", imgCircles);
	imwrite("circles.png", detectImgCircles);
	waitKey(0);

	return 0;
}
```

<img src="{{ '/pic/circles.png' | relative_url }}" style="zoom:50%;" />