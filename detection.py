#!\bin\python2.7

import numpy as np


class Detection(object):

    """ This class represents a bounding box detection in a single image.
    The box is represented in the format [x1, y1, x2, y2, confidence] """

    def __init__(self, pos, confidence, features=None):
        """
        :param pos (ndarray): Bounding box in format [x1, y1, x2, y2]
        :param confidence (float): Detector confidence score.
        :param feature (Optional[ndarray]): A feature vector that describes the object contained in this
        """
        self.pos = pos
        self.confidence = confidence
        self.features = features

    def unify_pos_score(self):
        """ Method for helping the object act more like an ndarray for ease of implementation """
        return np.concatenate((self.pos, self.confidence), axis=None)

    def __repr__(self):
        return str(self.unify_pos_score())

    def __str__(self):
        return self.unify_pos_score()

    def __len__(self):
        return len(self.unify_pos_score())

    def __getitem__(self, sliced):
        return self.unify_pos_score()[sliced]












