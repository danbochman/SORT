from collections import OrderedDict
from metric import Metric
from track import KalmanTrack
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np


class Tracker:

    def __init__(self, metric=None, matching_threshold=None, max_disappeared=4):
        """ initialize the next unique object ID along with two ordered dictionaries,
        used to keep track of mapping a given object ID to its centroid and
        number of consecutive frames it has been marked as "disappeared", respectively
        :param metric: (class Object) Metric class, determines metric used for distance matrix
        :param matching_threshold: (float) minimum value acceptable for IoU matching
        :param max_disappeared: (int) consecutive frames allowed before deletion """
        self.nextTrackID = 0
        self.matching_threshold = matching_threshold
        self.tracked = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.metric = Metric(metric)

    def deregister(self, object_id):
        """ to de-register an object ID we delete the object ID from
        both of our respective dictionaries
        :param object_id: (int) unique object id """
        del self.tracked[object_id]
        del self.disappeared[object_id]

    def handle_no_detections(self):
        """ This method is called when there are no new detections for the detector.
        loops over all the tracked objects, updates their 'disappeared counter',
        and deletes objects if their counter is expired """
        for track_id in self.disappeared.keys():
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self.deregister(track_id)
        return self.tracked


class KalmanTracker(Tracker):

    def __init__(self, metric='iou', matching_threshold=0.2):
        Tracker.__init__(self, metric, matching_threshold)

    def register(self, state):
        """ when registering an object we use the next available object
        ID to store the state
        :param state: (array) estimated state vector of detection """
        self.tracked[self.nextTrackID] = KalmanTrack(state)
        self.disappeared[self.nextTrackID] = 0
        self.nextTrackID += 1

    def update(self, detections):
        """Core method of the tracker, updates tracked objects after receiving detections.
        Associates detection with existing tracked objects, deletes disappeared detection and creates
        new trackers if non are associated with detections.
        :param detections: (array) list of detections (bounding boxes)
        :return self.objects: (array) list of tracked objects"""
        if len(detections) == 0:  # Check to see if there are no detections
            return self.handle_no_detections()

        # if we are currently not tracking any objects, register all the input centroids
        if len(self.tracked) == 0:
            for i in range(0, len(detections)):
                self.register(detections[i])

        else:
            self.associate(detections)

        # return the corrected estimate of tracked objects and their ids (for specific color)
        return [(ID, track.project()) for ID, track in self.tracked.items()]

    def associate(self, detections):
        """ Performs the Hungarian algorithm and assignment between detections and existing
        trackers according to distance matrix
        :param detections: (array) list of new detections """
        # Grab the set of object IDs and corresponding states
        track_ids = list(self.tracked.keys())
        # Get predicted tracked object states from Kalman Filter
        tracked_states = [track.predict() for track in self.tracked.values()]

        # Compute the distance matrix between detections and trackers according to metric
        D = self.metric.distance_matrix(tracked_states, detections)

        # Associate detections to existing trackers according to distance matrix
        rows, cols = linear_sum_assignment(-D)
        used_rows = set()
        used_cols = set()
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or column value before - ignore it
            if row in used_rows or col in used_cols:
                continue

            # Validate that the matching is above the threshold
            elif D[row, col] > self.matching_threshold:
                # update tracker - set new state and reset 'disappeared' counter
                track_id = track_ids[row]
                self.tracked[track_id] = self.tracked[track_id].update(detections[col])
                self.disappeared[track_id] = 0
                used_rows.add(row)
                used_cols.add(col)

        # Retrieve both the row and column indices we have NOT yet examined
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        # if the number of tracked objects >= the number of detections
        # check and see if some of these objects have potentially disappeared
        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                object_id = track_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        # otherwise, the number of detections > tracked objects - register each new detection
        else:
            for col in unused_cols:
                self.register(detections[col])


class ORBTracker(Tracker):

    def __init__(self, metric='FLANN', matching_threshold=0.2):
        Tracker.__init__(self, metric, matching_threshold)

    def register(self, state):
        """ when registering an object we use the next available object
        ID to store the state
        :param state: (array) estimated state vector of detection """
        self.tracked[self.nextTrackID] = KalmanTrack(state)
        self.disappeared[self.nextTrackID] = 0
        self.nextTrackID += 1

    def update(self, frame, detections):
        """Core method of the tracker, updates tracked objects after receiving detections.
        Associates detection with existing tracked objects, deletes disappeared detection and creates
        new trackers if non are associated with detections.
        :param detections: (array) list of detections (bounding boxes images)
        :return self.objects: (array) list of tracked objects"""
        if len(detections) == 0:  # Check to see if there are no detections
            return self.handle_no_detections()

        # if we are currently not tracking any objects, register all the input centroids
        if len(self.tracked) == 0:
            for i in range(0, len(detections)):
                self.register(detections[i])

        else:
            self.associate(frame, detections)

        # return the corrected estimate of tracked objects and their ids (for specific color)
        return [(ID, track.project()) for ID, track in self.tracked.items()]

    def associate(self, frame, detections):
        """ Performs the Hungarian algorithm and assignment between detections and existing
        trackers according to distance matrix
        :param frame: (np.array) new frame to be analyzed
        :param detections: (array) list of new detections """
        # Grab the set of object IDs and corresponding states
        track_ids = list(self.tracked.keys())
        # Get predicted tracked object states from Kalman Filter
        tracked_states = [track.predict() for track in self.tracked.values()]
        tracked_boxes = ORBTracker.crop_bbox_from_frame(frame, tracked_states)
        detections_boxes = ORBTracker.crop_bbox_from_frame(frame, detections)

        # Compute the distance matrix between detections and trackers according to metric
        D = self.metric.distance_matrix(tracked_boxes, detections_boxes)

        # Associate detections to existing trackers according to distance matrix
        rows, cols = linear_sum_assignment(-D)
        used_rows = set()
        used_cols = set()
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or column value before - ignore it
            if row in used_rows or col in used_cols:
                continue

            # Validate that the matching is above the threshold
            elif D[row, col] > self.matching_threshold:
                # update tracker - set new state and reset 'disappeared' counter
                track_id = track_ids[row]
                self.tracked[track_id] = self.tracked[track_id].update(detections[col])
                self.disappeared[track_id] = 0
                used_rows.add(row)
                used_cols.add(col)

        # Retrieve both the row and column indices we have NOT yet examined
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        # if the number of tracked objects >= the number of detections
        # check and see if some of these objects have potentially disappeared
        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                object_id = track_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        # otherwise, the number of detections > tracked objects - register each new detection
        else:
            for col in unused_cols:
                self.register(detections[col])

    @staticmethod
    def crop_bbox_from_frame(frame, bboxes):
        bboxes_crop = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype('int')
            if x2 >= frame.shape[1]:
                x2 = frame.shape[1] - x1
            elif y2 >= frame.shape[0]:
                y2 = frame.shape[0] - y1
            if isinstance(x1, np.ndarray):
                bboxes_crop.append(frame[y1[0]:y2[0], x1[0]:x2[0]])
            else:
                bboxes_crop.append(frame[y1:y2, x1:x2])

        return bboxes_crop






