---
layout: default
title: Stereo Vision
nav_order: 12
permalink: /stereo-vision/
description: Stereo pipeline overview: calibration, epipolar geometry, disparity, and depth.
---

- Reconstructing 3D geometry based on camera images from two or more viewpoints
- Human Vision

How to estimate the distance using stereo vision?

- Calibrate the cameras (Intrinsic and extrinsic calibration)
- Create an epipolar scheme using epipolar geomatry
- Build a disparity and a depth map

Depth map will be combined with an obstacle detection algorithm

## Time of Flight Camera

- Estimate distance by measuring the time of flight of the light signal between the camera and the subject for each point of the image

<img src="{{ '/pic/flight.png' | relative_url }}" style="zoom:38%;" />