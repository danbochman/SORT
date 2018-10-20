#!\bin\python2.7

"""
A module for testing SORT operation with the MOT benchmark
"""

from __future__ import print_function
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from tracker import Tracker
import numpy as np
from sort import Sort


class SortTest(Sort):

    def __init__(self, seq):
        """ Sets key parameters for SortTest """
        self.tracker = Tracker()
        self.seq = seq
        self.start_tracking()

    @staticmethod
    def show_source(seq, frame, phase='train'):
        """ Method for displaying the origin video being tracked """
        return io.imread('mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame))

    @staticmethod
    def check_data_path():
        ''' Validates correct implementation of symbolic link to data for SORT '''
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

        file_path = 'data/%s/det.txt' % self.seq
        self.load_detections(file_path)
        colours = np.random.rand(32, 3)
        plt.ion()
        fig = plt.figure()

        for frame_idx in range(1, int(self.detections[:, 0].max())):
            new_detections = self.detections[self.detections[:, 0] == frame_idx, 2:7]
            new_detections[:, 2:4] += new_detections[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

            ax1 = fig.add_subplot(111, aspect='equal')
            im = SortTest.show_source(self.seq, frame_idx)
            ax1.imshow(im)
            plt.title(self.seq + ' Tracked Targets')
            ids_and_tracks = self.tracker.update(new_detections[:, :4])

            # Draw bounding boxes
            for ID, bbox in ids_and_tracks:
                b = bbox.astype(np.int32)
                ax1.add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, lw=3,
                                                ec=colours[ID % 32, :]))
                ax1.set_adjustable('box-forced')

            # Show tracked frame
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

        plt.close(fig)


def main():
    '''Starts the tracker on source video'''
    # Initialize the parameters for SORT
    sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2',
                 'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2']
    SortTest.check_data_path()

    for seq in sequences:
        mot_tracker = SortTest(seq)
        del mot_tracker


if __name__ == '__main__':
    main()




