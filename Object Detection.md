# Object Detection

## Applications of Object Detection

- Object Tracking
- Security Systems
- Autonomous Vehicles
- Etc

## Haar Classifier

- Haar Features in images
  - Efficient calculation

- Training the classifier with images
  - Both positive and negative samples

- Selects features during training to create strong classifiers
- Weak and Strong Classifiers
  - Adaboost to make it more efficient and having less features

### Haar - Features

- Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle
- Specific feature depends on what we want to detect
- How can we select the best features?
  - Adaboost - finds the best threshold that classifies the faces to positive and negative
  - Is used to find the best features and reduce them

**Adaboost: Aadptive Boosting**

**Detection != Recognition**

```C++
#include<opencv2/objdetect.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

void faceDetection(Mat frame) {
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	// Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	for (size_t i = 0; i < faces.size(); i++) {
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 6);
		Mat faceROI = frame_gray(faces[i]);
	}

	imshow("Live Face Detection", frame);
}

int main() {
	string faceClassifier = "../data/haar/haarcascade_frontalface_alt2.xml";
	if (!face_cascade.load(faceClassifier)) {
		cout << "Could not load the classifier";
		return -1;
	}

	cout << "Loaded successfully" << endl;
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "Could not open video capture";
		return -1;
	}
	Mat frame;
	while (capture.read(frame)) {
		if (frame.empty()) {
			cout << "No frame captured from camera";
			break;
		}

		// Apply the face detection with the haar cascade classifier
		faceDetection(frame);
		if (waitKey(10) == 'q') {
			break; // Terminate program if q pressed
		}
	}
	return 0;
}
```

