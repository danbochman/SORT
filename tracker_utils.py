import numpy as np


def xysr_to_xxyy(x, score=None):
    """ Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape(4, 1)
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score])


def xxyy_to_xysr(bbox):
    """ Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio"""
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h
    r = w/float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def bbox_to_centroid(detections):
    input_centroids = np.zeros((len(detections), 2), dtype="int")
    for (i, (startX, startY, endX, endY)) in enumerate(detections):
        # use the bounding box coordinates to derive the centroid
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        input_centroids[i] = (cX, cY)

    return input_centroids

