#!\bin\python2.7

from __future__ import division, print_function
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from reid_nn.reidnn import reid


class Metric:
    """
    class for containing metric configurations intended for use for trackers.
    initialized by calling it with the desired metric (default: 'iou'), contains static methods for
    calculating distance matrices between detections and tracks.
    """

    def __init__(self, metric='iou'):
        """
        :param metric (str): Switch between different metrics ('iou', 'euc', 'FLANN')
        """
        self.metric = metric

    def distance_matrix(self, tracks, detections):
        """
        Compute distance between detections and tracks.
        Utilizes the scipy.spatial.distance.cdist for computation acceleration where possible.
        In cases where detection\trackers are 3D arrays, use the staticmethod mdist instead.
        Returns a cost matrix of shape len(detections), len(trackers).
        where element (i, j) contains the closest squared distance between detections[i] and trackers[j].
        :param detections: A list with length M of detections objects (can be 2D or 3D arrays)
        :param trackers: A list with length N targets to match the given trackers against (an be 2D or 3D arrays)
        :return: MxN ndarray distance matrix between detections and trackers
        """
        if self.metric == 'iou':
            # Make sure detection are in the right format for operation
            tracks = np.array(tracks)[:, :, 0]
            return cdist(tracks, detections, Metric.iou)

        if self.metric == 'ORB':
            # Calls the mdist static method since in this case detections and tracks are lists of images (3D arrays)
            return Metric.mdist(tracks, detections, Metric.ORB)

        if self.metric == 'ReIDNN':
            # Calls the mdist static method since in this case detections and tracks are lists of images (3D arrays)
            return Metric.batch_mdist(tracks, detections, reid)

        if self.metric == 'euc':
            return cdist(tracks, detections)

    @staticmethod
    def mdist(arr1, arr2, func):
        """
        function for computing a more general distance matrix, where the inputs can be any type (e.g 3D matrices)
        :param arr1: (any type) this function is mostly relevant for images. arr1 would be new detections represented as images
        :param arr2: (any type) arr2 would be images associated with tracks.
        :param func: metric function for distance matrix
        :return: (np.ndarray) distance matrix
        """
        # initialize the distance matrix dimensions
        dm = np.zeros((len(arr1), len(arr2)))
        for i in xrange(len(arr1)):
            for j in xrange(len(arr2)):
                dm[i, j] = func(arr1[i], arr2[j])
        return dm

    @staticmethod
    def preprocess_images(tracks, detections):
        """
        static method for preparing the image crops from detections and tracks to be sent to the Re-ID NN
        :param detections, tracks: (array) list of images cropped from frame
        :return: (array) list of lists when each list is a pair of [detection, track] cropped images
        """
        img_w = 60
        img_h = 160
        detections = np.array(detections)
        tracks = np.array(tracks)
        detections = [detection if not (0 in detection.shape) else np.zeros((160, 60, 3)) for detection in detections]
        tracks = [track if not (0 in track.shape) else np.zeros((160, 60, 3)) for track in tracks]
        image_pairs = [[cv2.resize(track, (img_w, img_h)),
                        cv2.resize(detection, (img_w, img_h))]
                       for track in tracks for detection in detections]
        image_pairs = np.transpose(image_pairs, (1, 0, 2, 3, 4))
        return image_pairs

    @staticmethod
    def batch_mdist(tracks, detections, nn):
        """
        function for computing a more general distance matrix, where the inputs can be any type (e.g 3D matrices)
        :param tracks, detections: (arrays) list of cropped images from frame associated with tracks and detections
        :param nn: neural network for producing similarity score
        :return: (np.ndarray) distance matrix
        """
        image_pairs = Metric.preprocess_images(tracks, detections)
        predictions = nn(image_pairs)
        distance_matrix = predictions.reshape(len(tracks), len(detections))
        return distance_matrix

    @staticmethod
    def iou(boxA, boxB):
        """
        Return the intersection over union value between two bounding boxes
        The bounding boxes should be in format [xmin, ymin, xmax, ymax]
        :param boxA: (np.ndarray) bounding box
        :param boxB: (np.ndarray) bounding box
        :return: Intersection over Union score for the two bounding boxes inputs
        """
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

    @staticmethod
    def ORB(img1, img2):
        """
        Fast Approximate Nearest Neighbor Search implementation in OpenCV.
        Returns a score for image similarity between 0-1 (1 - Very similar)
        :param img1, img2: (ndarray) two image slices taken from frame representing bounding boxes
        :return: (float) scalar similarity score
        """
        # Initialize ORB feature detector with recommended parameters (may need tweaks according to use case)
        patchSize = 7
        orb = cv2.ORB_create(edgeThreshold=7, patchSize=patchSize, nlevels=8, scaleFactor=1.2, WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500, fastThreshold=20)

        # make sure the images shape are larger than the patchSize in ORB parameters
        shapes = [img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]]
        for shape in shapes:
            if shape < patchSize:
                return 0

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if isinstance(des1, type(None)) or isinstance(des2, type(None)):
            return 0

        # consider the appropriate reference for keypoint matches
        num_kp = min(len(kp1), len(kp2))

        # create BFMatcher object and match descriptors
        bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # ratio test as per Lowe's paper
        good_matches = []
        for match in matches:
            if len(match) != 2:
                continue
            m, n = match
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)

        score = len(good_matches) / num_kp
        # Uncomment the following lines if you want to see the similarity score given to 2 input images
        # cv2.imshow('img1', img1)
        # cv2.imshow('img2', img2)
        # print(score)
        # cv2.waitKey(0)
        return score









