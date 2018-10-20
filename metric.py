#!\bin\python2.7

from scipy.spatial.distance import cdist
import numpy as np


class Metric:
    """ For each target, returns matrix of distances between detections and trackers """
    def __init__(self, metric='iou'):
        """
        :param metric (str): Switch between different metrics
        :param matching_threshold (float): The matching threshold. Larger distance considered an invalid match.
        """
        self.metric = metric

    def distance_matrix(self, detections, trackers):
        """
        Compute distance between detections and trackers.
        Returns a cost matrix of shape len(detections), len(trackers).
        where element (i, j) contains the closest squared distance between detections[i] and trackers[j].
        :param detections: A list with length M of detections objects
        :param trackers: A list with length N targets to match the given trackers against
        :return: MxN ndarray distance matrix between detections and trackers
        """
        if self.metric == 'iou':
            # Make sure detection are in the right format for operation
            detections = np.array(detections)[:, :, 0]
            return cdist(detections, trackers, Metric.iou)

        if self.metric == 'euc':
            return cdist(detections, trackers)

    @staticmethod
    def iou(boxA, boxB):
        """ Return the intersection over union value between two bounding boxes
        The bounding boxes should be in format [xmin, ymin, xmax, ymax] """

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou






