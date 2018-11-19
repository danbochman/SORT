# SORT

This repository is meant to perform as a modular and flexible SORT (Simple Online Real-Time Tracking) algorithm.
The project is highly object oriented for easy future customization in mind.

The SORT cores:

sort.py
-------
This is the main app module which constructs a SORT object which holds: a video source, a specific tracker type, (optional) detector

tracker.py
----------
Contains the Tracker master class and a few specialized tracker which inherit from it, the trackers differ from each other by metric method by which they compare detections (e.g. IoU, features, Neural network)
The tracker hold all the 'Tracks' (also class objects) and their states for comparison with new detections

track.py
--------
Each individual detection is assigned to a Track object and registered by the tracker. The Track is an improved state estimator which utilizes a Kalman Filter to give more robust predictions to where each individual tracked object is expected to be between consecutive frames 


Great open source references which helped in the creation of this repository
--------------------------------------------------------------------------
A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences. By Alex Bewley
https://github.com/abewley/sort

Simple object tracking with OpenCV by Adrian Rosebrock
https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv

Tensorflow implementation of "An Improved Deep Learning Architecture for Person Re-Identification"
https://github.com/digitalbrain79/person-reid

Real-time Human Detection in Computer Vision
https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6



