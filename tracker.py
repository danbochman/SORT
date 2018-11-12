from collections import OrderedDict
from metric import Metric
from track import KalmanTrack
from scipy.optimize import linear_sum_assignment
import numpy as np


class Tracker:
    """
    Parent class for the general Tracker case, intended for creating the basis for inheritance for specialized trackers.
    Assumes the default use of KalmanFilter to assist tracking.
    """
    def __init__(self, metric=None, matching_threshold=None, max_disappeared=10):
        """ initialize the next unique object ID along with two ordered dictionaries,
        used to keep track of mapping a given object ID to its centroid and
        number of consecutive frames it has been marked as "disappeared", respectively
        :param metric: (class Object) Metric class, determines metric used for distance matrix
        :param matching_threshold: (float) minimum value acceptable for distance matrix matching
        :param max_disappeared: (int) consecutive frames not seen allowed before track deletion

        --- attributes ---
        nextTrackID (int) - Unique key generator for tracks
        tracked (OrderedDict) data structure for holding tracks (track ID,: Track class object)
        disappeared (OrderedDict) data structure for keeping coutn of how many frames each track disappeared
                    (track ID: disappearance count)
        """
        self.nextTrackID = 0
        self.matching_threshold = matching_threshold
        self.tracked = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.metric = Metric(metric)

    def register(self, state):
        """ Registers a new track with a new detection unmatched to other tracks
        assigns the state to a new Track class object and initialized its data structures
        :param state: (np.ndarray) initial state vector of object from new detection """
        self.tracked[self.nextTrackID] = KalmanTrack(state)
        self.disappeared[self.nextTrackID] = 0
        self.nextTrackID += 1

    def deregister(self, object_id):
        """ de-registers an object ID by deleting the object ID from both respective dictionaries
        :param object_id: (int) unique object id """
        del self.tracked[object_id]
        del self.disappeared[object_id]

    def handle_no_detections(self):
        """ This method is called when there are no new detections for the detector.
        loops over all the tracked objects, updates their 'disappeared counter',
        and deletes objects if their counter is expired
         :return (OrderedDict) dictionary containing (track_id: Track)
         """
        for track_id in self.disappeared.keys():
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self.deregister(track_id)
        return self.tracked

    def project(self):
        tracks = [(ID, track.project()) for ID, track in self.tracked.items()]

        # Make sure tracks are unpacked to their respective states
        if isinstance(tracks, OrderedDict):
            tracks = [(ID, track.project()) for ID, track in tracks.items() if self.disappeared[ID] == 0]

        return tracks

    def linear_assignment(self, D, track_ids, detections):
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

        # Retrieve both the row and column indices we have not yet examined
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

    def reset(self):
        """ resets the tracker. deletes all registered tracks. relevant for transferring the track to new video """
        self.nextTrackID = 0
        self.tracked = OrderedDict()
        self.disappeared = OrderedDict()

    @staticmethod
    def crop_bbox_from_frame(frame, bboxes):
        """
        static method used by the ORBTracker for cropping slices from frame according to bounding box
        :param frame: (array) np.ndarray representing image frame associated with current detections
        :param bboxes: (array) list of bounding boxes
        :return: (array) list where each element is an image crop from frame according to bounding box
        """
        bboxes_crop = []
        for bbox in bboxes:
            # Make sure the coordinates are type int so they can function as indices
            x1, y1, x2, y2 = bbox.astype('int')

            # Truncate coordinates of detections if value exceeds frame image dimensions
            if x2 >= frame.shape[1]:
                x2 = [frame.shape[1] - 1]
            elif y2 >= frame.shape[0]:
                y2 = [frame.shape[0] - 1]

            # Handle bounding boxes coordinates retrieved as list (not scalar)
            if isinstance(x1, np.ndarray):
                bboxes_crop.append(frame[y1[0]:y2[0], x1[0]:x2[0]])
            else:
                bboxes_crop.append(frame[y1:y2, x1:x2])

        return bboxes_crop


class KalmanTracker(Tracker):
    """
    Specialized tracker class which inherits from the basic Tracker class
    Utilizes the KalmanFilter and the IoU metric for more robust bounding box associations
    """
    def __init__(self, metric='iou', matching_threshold=0.2):
        """ Initialize the tracker from base class with relevant metrics """
        Tracker.__init__(self, metric, matching_threshold)

    def update(self, detections):
        """
        Core method of the tracker, updates tracked objects after receiving detections.
        Associates detection with existing tracked objects, deletes disappeared detection and creates
        new trackers if non are associated with detections.
        :param detections: (array) list of detections (bounding boxes in [x1,x2,y1,y2] format)
        :return (array) list of tracked objects tuples in format (ID, [x1,x2,y1,y2])
        """
        # Check to see if there are no detections and handle if so
        if len(detections) == 0:
            return self.handle_no_detections()

        # if we are currently not tracking any objects, register all detections to new tracks
        if len(self.tracked) == 0:
            for i in range(0, len(detections)):
                self.register(detections[i])

        # otherwise, check to see how the new detections relate to current tracks
        else:
            self.associate(detections)

        # return the corrected estimate of tracked objects and their unique ids
        return self.project()

    def associate(self, detections):
        """
        Performs the Hungarian algorithm and assignment between detections and existing
        trackers according to distance matrix and metric specified
        updates the Tracker class dictionaries after associations
        :param detections: (array) list of new detections
         """
        # Grab the set of object IDs and corresponding states
        track_ids = list(self.tracked.keys())

        # Get predicted tracked object states from KalmanFilter
        tracked_states = [track.predict() for track in self.tracked.values()]

        # Compute the distance matrix between detections and trackers according to metric
        D = self.metric.distance_matrix(tracked_states, detections)

        # Associate detections to existing trackers according to distance matrix
        self.linear_assignment(D, track_ids, detections)


class ORBTracker(Tracker):
    """
    Specialized tracker class which inherits from the basic Tracker class
    Utilizes the KalmanFilter and feature matching (ORB) for more accurate bounding box associations
    """
    def __init__(self, matching_threshold=0.01):
        """ Initialize the tracker from base class with relevant metrics """
        Tracker.__init__(self, matching_threshold)
        self.metric_orb = Metric('ORB')
        self.metric_iou = Metric('iou')

    def update(self, frame, detections):
        """
        Core method of the tracker, updates tracked objects after receiving detections.
        Associates detection with existing tracked objects, deletes disappeared detection and creates
        new trackers if non are associated with detections.
        :param frame: (array) np.ndarray representing image frame associated with current detections
        :param detections: (array) list of detections (bounding boxes in [x1,x2,y1,y2] format)
        :return (array) list of tracked objects tuples in format (ID, [x1,x2,y1,y2])
        """
        # Check to see if there are no detections and handle if so
        if len(detections) == 0:
            return self.handle_no_detections()

        # if we are currently not tracking any objects, register all detections to new tracks
        if len(self.tracked) == 0:
            for i in range(0, len(detections)):
                self.register(detections[i])

        # otherwise, check to see how the new detections relate to current tracks
        else:
            self.associate(frame, detections)

        # return the corrected estimate of tracked objects and their unique ids
        return self.project()

    def associate(self, frame, detections):
        """
        Performs the Hungarian algorithm and assignment between detections and existing
        trackers according to distance matrix and metric specified.
        updates the Tracker class dictionaries after associations
        :param frame: (array) np.ndarray representing image frame associated with current detections
        :param detections: (array) list of new detections
         """
        # Grab the set of object IDs and corresponding states
        track_ids = list(self.tracked.keys())

        # Get predicted tracked object states from Kalman Filter
        tracked_states = [track.predict() for track in self.tracked.values()]

        # Crop the image from each bounding box associated to new detections and current tracks
        tracked_crops = Tracker.crop_bbox_from_frame(frame, tracked_states)
        detections_crops = Tracker.crop_bbox_from_frame(frame, detections)

        # Compute the distance matrix between detections and trackers according to metric
        D_orb = self.metric_orb.distance_matrix(tracked_crops, detections_crops)
        D_iou = self.metric_iou.distance_matrix(tracked_states, detections)
        w = 0.2
        D = np.multiply(w * D_orb, (1-w) * D_iou)

        # Associate detections to existing trackers according to distance matrix
        self.linear_assignment(D, track_ids, detections)


class ReIDTracker(Tracker):
    """
    Specialized tracker class which inherits from the basic Tracker class
    Utilizes the KalmanFilter and a person re-identification neural network for more accurate bounding box associations
    """
    def __init__(self, matching_threshold=0.2, diff_threshold=0.2):
        """ Initialize the tracker from base class with relevant metrics """
        Tracker.__init__(self, matching_threshold=matching_threshold)
        self.metric_nn = Metric('ReIDNN')
        self.metric_iou = Metric('iou')
        self.diff_threshold = diff_threshold

    def update(self, frame, detections):
        """
        Core method of the tracker, updates tracked objects after receiving detections.
        Associates detection with existing tracked objects, deletes disappeared detection and creates
        new trackers if non are associated with detections.
        :param frame: (array) np.ndarray representing image frame associated with current detections
        :param detections: (array) list of detections (bounding boxes in [x1,x2,y1,y2] format)
        :return (array) list of tracked objects tuples in format (ID, [x1,x2,y1,y2])
        """
        # Check to see if there are no detections and handle if so
        if len(detections) == 0:
            return self.handle_no_detections()

        # if we are currently not tracking any objects, register all detections to new tracks
        if len(self.tracked) == 0:
            for i in range(0, len(detections)):
                self.register(detections[i])

        # otherwise, check to see how the new detections relate to current tracks
        else:
            self.associate(frame, detections)

        # return the corrected estimate of tracked objects and their unique ids
        return self.project()

    def associate(self, frame, detections):
        """
        Performs the Hungarian algorithm and assignment between detections and existing
        trackers according to distance matrix and metric specified.
        updates the Tracker class dictionaries after associations
        :param frame: (array) np.ndarray representing image frame associated with current detections
        :param detections: (array) list of new detections
         """
        # Grab the set of object IDs and corresponding states
        track_ids = list(self.tracked.keys())

        # Get predicted tracked object states from Kalman Filter
        tracked_states = [track.predict() for track in self.tracked.values()]

        # Crop the image from each bounding box associated to new detections and current tracks
        tracked_crops = Tracker.crop_bbox_from_frame(frame, tracked_states)
        detections_crops = Tracker.crop_bbox_from_frame(frame, detections)

        # Compute the distance matrix between detections and trackers according to metric
        D = self.metric_iou.distance_matrix(tracked_states, detections)
        D_iou_sorted = -np.sort(-D)

        # Check if there are difficult overlapping IoU between a track and several detections
        for row_idx in xrange(D.shape[0]):
            if (D_iou_sorted[row_idx, 0] - D_iou_sorted[row_idx, 1] < self.diff_threshold) and (D_iou_sorted[row_idx, 0] > self.matching_threshold):
                # Consult with re-identification network
                D_nn = self.metric_nn.distance_matrix(tracked_crops, detections_crops)
                D = np.multiply(D, D_nn)
                break

        # Associate detections to existing trackers according to distance matrix
        self.linear_assignment(D, track_ids, detections)









