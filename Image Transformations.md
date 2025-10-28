---
layout: default
title: Image Transformations
nav_order: 8
permalink: /image-transformations/
description: Affine and perspective transforms, interpolation, and distortion models.
---

## Geometric Transformations

- Types of Geometric Transformations:
  - Geometric
  - Pixel coordinate
  - Brightness interpolation

- Applications
  - Computer graphics
  - Distortion - Introduce / Eliminate
  - Image processing / Preprocessing
  - Text recognition
  - Recognition of signs, numbers, etc

## Formulation of the problem

- Distorted image $f(i, j)$

- Corrected image $f'(i', j')$

- Mapping between the images
  $$
  i = T_i(i', j'),\quad j = T_j(i', j')
  $$

- Define the Transformation
  - Known in advance / determine through correspondences
  - Image to know
  - Image to image

- Apply the defined Transformation
  - For every point in the output image
  - Determine where it came from using T from mapping
  - Interpolate a value for the output point

## Transformations

- Linear Transformation in 3D
  - Affine Transformation
    - Euclidean ( 6 DoF ) ---- 3 for translation, 3 for rotation
- Higher Order Transformations
  - Parameterized
    - B-splines
  - Freedom
    - Warp field

### Affine Transformations

- Known Transformations
  - Translation, Rotation

- Unknown Transformations
  - Would require at least 3 observations
  - Could be points in an image
    - Corners of objects

### Unknown Affine Transformation

- More observations - We need at least 3
  - Better estimate of the coefficients

- Uses pseudo inverse
  - For unknown transformations

## Perspective Transformations

- Perspective projection
- Planar surface
- Not parallel to the image plane
- Can't be corrected with the affine trans
- Therefore we need a perspective trans ---- 透视变换

## Rectifying Homographs

- Image transformations can be computed such that scan lines can be directly matched on images

纠正Homograph

## Brightness Interpolation

- Location are not integer coordinates
- Interpolate output pixel value from the nearby pixels in the original image
- Interpolation methods
  - Nearest neighbor
  - Bilinear interpolation
  - Bicubic interpolation

## Distortion - Camera Models

- Calibrate using multiple images
- Calibrate with known objects
- Change positions
- Compute the camera matrix and distortion parameters
- Remove distortion using the distortion parameters

<hr/>

```cpp
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "mouse.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    // Read image from file 
    Mat snookerImg = imread("../data/snooker.jpg");
    // Mat plateImg = imread("../data/plate.jpg");
    Mat perspecImg, affineImg;

    //if fail to read the image
    /*if (snookerImg.empty() || plateImg.empty())
    {
        cout << "Error loading the image" << endl;
        return -1;
    }*/

    //Create a window
    //namedWindow("Original Plate", 1);
    namedWindow("Original Snooker", 1);

    // Find points in image with mouse
    //setMouseCallback("Original Plate", platePoints, NULL);
    setMouseCallback("Original Snooker", snookerPoints, NULL);

    //imshow("Original Plate", plateImg);
    imshow("Original Snooker", snookerImg);

    waitKey(0);

    /*cout << "Points Plate: " << endl;
    for (auto& i : pointsPlate) {
        cout << "( " << i.x << ", " << i.y << " )" << endl;
    }*/

    cout << "Destination Plate: " << endl;
    for (auto& i : destinationPlate) {
        cout << "( " << i.x << ", " << i.y << " )" << endl;
    }


    // Affine Transforamtion of number plate
    // Find points from image and distination points
    //vector<Point2f> sourcePlate, destinationPlate;
    //sourcePlate = { Point2f(400,250), Point2f(400,320), Point2f(200,330) };
    //destinationPlate = { Point2f(770,350), Point2f(770,450), Point2f(250,370) };


    // Calculate the affine matrix from the found points in the image
    //Mat affineMatrix = getAffineTransform(pointsPlate, destinationPlate);
    // Apply the affine transformation on the image
    //warpAffine(plateImg, affineImg, affineMatrix, plateImg.size());

    //show the image
    //imshow("Plate Transformation", affineImg);



    // Perspective Transformation of snooker table
    // Find points from image and distination points
    //vector<Point2f> sourceSnooker, destinationSnooker;
    //sourceSnooker = { Point2f(338,645), Point2f(671,650), Point2f(922,916), Point2f(101,919) };
    //destinationSnooker = { Point2f(278,223), Point2f(785,220), Point2f(830,905), Point2f(205,907) };


    // Calculate the perspective matrix from the found points in the image
    Mat perspectiveMatrix = getPerspectiveTransform(pointsSnooker, destinationSnooker);
    // Apply the perspective transformatin on the image
    warpPerspective(snookerImg, perspecImg, perspectiveMatrix, snookerImg.size());

    // Show image
    imshow("Perspective Transformation", perspecImg);


    // Wait until user press some key
    waitKey(0);

    return 0;

}
```

```cpp
#pragma once
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

vector<Point2f> pointsPlate;
vector<Point2f> destinationPlate;
vector<Point2f> pointsSnooker;
vector<Point2f> destinationSnooker;

void platePoints(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN and pointsPlate.size() < 3)
    {
        cout << "Plate - position (" << x << ", " << y << ")" << endl;
        pointsPlate.push_back(Point2f(x, y));
    }
    else if (event == EVENT_LBUTTONDOWN) {
        cout << "Plate - destination (" << x << ", " << y << ")" << endl;
        destinationPlate.push_back(Point2f(x, y));
    }
}

void snookerPoints(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN and pointsSnooker.size() < 4)
    {
        cout << "Snooker - position (" << x << ", " << y << ")" << endl;
        pointsSnooker.push_back(Point2f(x, y));
    }
    else if (event == EVENT_LBUTTONDOWN) {
        cout << "Snooker - destination (" << x << ", " << y << ")" << endl;
        destinationSnooker.push_back(Point2f(x, y));
    }
}
```

