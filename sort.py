#!\bin\python2.7

"""
Main module for the real-time tracker class execution. Based on the SORT algorithm
"""

from __future__ import print_function
import os.path
from tracker import KalmanTracker, ORBTracker, ReIDTracker
import numpy as np
import cv2
from tracker_utils import bbox_to_centroid
import random
import colorsys


class SORT:

    def __init__(self, seq=None, tracker='ReID', mode='benchmark'):
        """
         Sets key parameters for SORT
        :param seq: (string) relevant for 'benchmark' mode. name of folder containing sequences of images
        :param tracker: (string) 'ORB', 'Kalman' or 'ReID', determines which Tracker class will be used for tracking
        :param mode: (string) 'benchmark' or 'stream', determines what the video source will be for SORT to track

        ---- attributes ---
        detections (list) - relevant for 'benchmark' mode, data structure for holding all the detections from file
        frame_count (int) - relevant for 'benchmark' mode, frame counter, used for indexing and looping through frames
        """
        if tracker == 'Kalman':
            self.tracker = KalmanTracker()
        if tracker == 'ORB':
            self.tracker = ORBTracker()
        if tracker == 'ReID':
            self.tracker = ReIDTracker()
        self.detections = None
        self.mode = mode
        self.seq = seq
        if self.mode == 'benchmark':
            # Load pre-made detections for .txt file (from MOT benchmark)
            SORT.check_data_path()
            file_path = 'data/%s/det.txt' % self.seq
            self.detections = np.loadtxt(file_path, delimiter=',')
        self.frame_count = 1
        self.start_tracking()

    def next_frame(self):
        """
        Method for handling the correct way to fetch the next frame according to the 'mode' attribute applied
        :return: (np.ndarray) next frame, (np.ndarray) detections for that frame
        """
        if self.mode == 'benchmark':
            frame = SORT.show_source(self.seq, self.frame_count)
            new_detections = self.detections[self.detections[:, 0] == self.frame_count, 2:7]
            new_detections[:, 2:4] += new_detections[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            self.frame_count += 1
            return frame, new_detections[:, :4]

    def start_tracking(self):
        """
        Main driver method for the SORT class, starts tracking detections from source.
        Receives list of associated detections for each frame from its tracker (Kalman or ORB),
        Shows the frame with color specific bounding boxes surrounding each unique track.
        """
        while True:
            # Fetch the next frame from video source, if no frames are fetched, stop loop
            frame, detections = self.next_frame()
            if frame is None:
                break

            # Send new detections to set tracker
            if isinstance(self.tracker, KalmanTracker):
                tracks = self.tracker.update(detections)
            elif isinstance(self.tracker, ORBTracker) or isinstance(self.tracker, ReIDTracker):
                tracks = self.tracker.update(frame, detections)
            else:
                raise Exception('[ERROR] Tracker type not specified for SORT')

            # Look through each track and display it on frame (each track is a tuple (ID, [x1,y1,x2,y2])
            for ID, bbox in tracks:
                # Generate pseudo-random colors for bounding boxes for each unique ID
                random.seed(ID)

                # Make sure the colors are strong and bright and draw the bounding box around the track
                h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
                color = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
                startX, startY, endX, endY = bbox.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              color, 2)

                # Calculate centroid from bbox, display it and its unique ID
                centroid = bbox_to_centroid(bbox)
                text = "ID {}".format(ID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

            # Show tracked frame
            cv2.imshow("Video Feed", frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    @staticmethod
    def show_source(seq, frame, phase='train'):
        """ Method for displaying the origin video being tracked """
        return cv2.imread('mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame))

    @staticmethod
    def check_data_path():
        """ Validates correct implementation of symbolic link to data for SORT """
        if not os.path.exists('mot_benchmark'):
            print('''
            ERROR: mot_benchmark link not found!\n
            Create a symbolic link to the MOT benchmark\n
            (https://motchallenge.net/data/2D_MOT_2015/#download)
            ''')
            exit()


def main():
    """ Starts the tracker on source video. Can start multiple instances of SORT in parallel """
    # sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2',
    #              'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2']
    sequences = ['PETS09-S2L1']
    for seq in sequences:
        mot_tracker = SORT(seq)
        del mot_tracker


if __name__ == '__main__':
    main()
