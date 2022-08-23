# Background Subtraction

- Pre-processing of the image
- Segmentation of objects
- Remove background of apply filters
- How to do Background Subtraction?
  - Subtracting two images from each other
  - K Nearest Neighbor
  - Mixture of Gaussian
  - Neural Networks

```C++
#include<opencv2/objdetect.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main() {
	Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2();
	// Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorKNN();
	VideoCapture capture = VideoCapture(0, CAP_DSHOW);

	if (!capture.isOpened()) {
		cout << "Unabled to open";
		return -1;
	}
	Mat frame, fgMask;
	while (true) {
		capture >> frame;
		if (frame.empty()) {
			cout << "No frame captured from camera";
			break;
		}
		resize(frame, frame, { 800, 800 });
		backSub->apply(frame, fgMask);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(fgMask, fgMask, kernel, Point(-1, -1), 2);
		dilate(fgMask, fgMask, kernel, Point(-1, -1), 2);

		rectangle(frame, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();

		putText(frame, frameNumberString.c_str(), Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27) {
			break; // Terminate program if q pressed
		}
	}
	return 0;
}
```

## FPS Count

```C++
#include<opencv2/opencv.hpp>
#include<iostream>
#include<time.h>

using namespace std;
using namespace cv;

int main() {
	// Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorKNN();
	VideoCapture capture(0);

	double fps = capture.get(CAP_PROP_FPS);
	cout << "Frame per second camera: " << fps << endl;

	int num_frames = 1;
	clock_t start;
	clock_t end;
	cout << "Capturing " << num_frames << " frames" << endl;
	double ms, fpsLive;
	if (!capture.isOpened()) {
		cout << "Unabled to open";
		return -1;
	}
	Mat frame, fgMask;
	while (true) {
		start = clock(); // Starg time
		capture >> frame;
		if (frame.empty()) {
			cout << "No frame captured from camera";
			break;
		}
		resize(frame, frame, { 800, 800 });
		backSub->apply(frame, fgMask);
		erode(fgMask, fgMask, (5, 5));
		dilate(fgMask, fgMask, (5, 5));

		long sum = 0;
		int N = 1000;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				sum += 1;
			}
		}
		end = clock();
		double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
		cout << "Times taken: " << seconds << "seconds" << endl;
		fpsLive = double(num_frames) / double(seconds);

		rectangle(frame, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();

		putText(frame, "FPS: " + to_string(fpsLive), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255));
		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27) {
			break; // Terminate program if q pressed
		}
	}
	return 0;
}
```

