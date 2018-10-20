#!\bin\python2.7

from kalman_filter import KalmanFilter
from tracker_utils import xxyy_to_xysr, xysr_to_xxyy
import numpy as np


class Track:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    """

    def __init__(self, initial_state, filter='Kalman'):
        """
        Initialises a tracked object according to initial bounding box.
        :param filter: (class Object) Filter class for enhancing association between detections and tracked objects
        :param initial_state: (array) single detection in form of bounding box [X_min, Y_min, X_max, Y_max]
        """
        if filter == 'Kalman':
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.init_kalman(initial_state)

    def init_kalman(self, initial_state):
        """ if Kalman Filter was selected for track, this method initializes the constant velocity
        model for tracking bounding boxes by default """
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = xxyy_to_xysr(initial_state)

    def project(self):
        return xysr_to_xxyy(self.kf.x)

    def update(self, new_detection):
        self.kf.update(xxyy_to_xysr(new_detection))
        return self

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

        return self.project()







