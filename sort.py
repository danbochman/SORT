#!\bin\python2.7
# USAGE
# python sort.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
"""
SORT: A Simple, Online and Real-Time Tracker
"""
from tracker import Tracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from tracker_utils import bbox_to_centroid
import sort_test


class Sort:
    def __init__(self, args, source='test', detector=False):
        self.args = args
        self.tracker = Tracker()
        self.H = None
        self.W = None
        if detector:
            self.detector = self.init_detector()
        if source == 'stream':
            self.init_stream()
        if source == 'test':
            sort_test.main()

    def init_detector(self):
        # load our serialized model from disk
        print("[INFO] loading model...")
        return cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])

    def init_stream(self):
        # initialize the video stream and allow the camera sensor to warmup
        print("[INFO] starting video stream...")

        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # read the next frame from the video stream and resize it
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # if the frame dimensions are None, grab them
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]

            """ construct a blob from the frame, pass it through the network,
            obtain our output predictions, and initialize the list of
            bounding box rectangles """
            blob = cv2.dnn.blobFromImage(frame, 1.0, (self.W, self.H), (104.0, 177.0, 123.0))
            self.detector.setInput(blob)
            detections = self.detector.forward()
            bboxes = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # filter out weak detections by threshold
                if detections[0, 0, i, 2] > self.args["confidence"]:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object, then update the bounding box rectangles list
                    bbox = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    bboxes.append(bbox.astype("int"))

                    # draw a bounding box surrounding the object
                    (startX, startY, endX, endY) = bbox.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # update our tracker using the computed set of bounding box rectangles
            objects = self.tracker.update(bboxes)

            # loop over the tracked objects
            for objectID, state in objects.items():
                # draw both the ID of the object and the centroid of the object on the output frame
                centroid = bbox_to_centroid(state)
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # Cleanup
        cv2.destroyAllWindows()
        vs.stop()


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=False,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    mot_tracker = Sort(args)


if __name__ == '__main__':
    main()


