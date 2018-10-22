#!\bin\python2.7

"""
A module for testing SORT operation with the MOT benchmark
"""

from __future__ import print_function
import os.path
from tracker import Tracker
import numpy as np
import cv2
from tracker_utils import bbox_to_centroid
import random
import colorsys


class SortTest:

    def __init__(self, seq):
        """ Sets key parameters for SortTest """
        self.tracker = Tracker()
        self.seq = seq
        self.start_tracking()

    @staticmethod
    def show_source(seq, frame, phase='train'):
        """ Method for displaying the origin video being tracked """
        return cv2.imread('mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame))

    @staticmethod
    def check_data_path():
        """ Validates correct implementation of symbolic link to data for SORT """
        if not os.path.exists('mot_benchmark'):
            print ('''
            ERROR: mot_benchmark link not found!\n
            Create a symbolic link to the MOT benchmark\n
            (https://motchallenge.net/data/2D_MOT_2015/#download)
            ''')
            exit()

    def load_detections(self, file_path, extension='txt'):
        """
        Load detections from source file
        :param file_path: (str) path to data
        :param extension: (str) data type (extension) default is 'txt'
        :return: (ndarray) detections
        """
        if extension == 'txt':
            self.detections = np.loadtxt(file_path, delimiter=',')

    def start_tracking(self):
        """
        Applies the SORT tracker on sequence input, plots image with bounding box for each frame
        """
        # Load pre-made detections for .txt file (from MOT benchmark)
        file_path = 'data/%s/det.txt' % self.seq
        self.load_detections(file_path)

        for frame_idx in range(1, int(self.detections[:, 0].max())):
            new_detections = self.detections[self.detections[:, 0] == frame_idx, 2:7]
            new_detections[:, 2:4] += new_detections[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            frame = SortTest.show_source(self.seq, frame_idx)
            ids_and_tracks = self.tracker.update(new_detections[:, :4])

            # Draw bounding boxes and centroids
            for ID, bbox in ids_and_tracks:
                # Generate pseudo-random colors for bounding boxes
                random.seed(ID)
                h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
                color = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
                startX, startY, endX, endY = bbox.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              color, 2)
                centroid = bbox_to_centroid(bbox)
                text = "ID {}".format(ID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

            # Show tracked frame
            cv2.imshow("Frame", frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


def main():
    """ Starts the tracker on source video """
    # Initialize the parameters for SORT
    sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2',
                 'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2']
    SortTest.check_data_path()

    for seq in sequences:
        mot_tracker = SortTest(seq)
        del mot_tracker


if __name__ == '__main__':
    main()




