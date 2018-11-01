import numpy as np


def xysr_to_xxyy(b, score=None):
    """
     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    :param b: (np.ndarray) bounding box in [x,y,s,r] format
    :param score: (float) Mostly irrelevant, incase bounding box comes with score attached to position
    :return: (np.ndarray) same bounding box in [x1,y1,x2,y2] format
    """
    w = np.sqrt(b[2]*b[3])
    h = b[2]/w
    if score is None:
        return np.array([b[0]-w/2., b[1]-h/2., b[0]+w/2., b[1]+h/2.]).reshape(4, 1)
    else:
        return np.array([b[0]-w/2., b[1]-h/2., b[0]+w/2., b[1]+h/2., score])


def xxyy_to_xysr(bbox):
    """
     Takes a bounding box in the form [x1,y1,x2,y2] and returns it in the form [x,y,s,r]
    :param bbox: (np.ndarray) bounding box in [x,y,s,r] format
    :return: (np.ndarray) same bounding box in [x,y,s,r] format
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h
    r = w/float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def bbox_to_centroid(bbox):
    """
    Computes centroid of bbox in format [xmin, xmax, ymin, ymax]
    :param bbox: (array) bounding box
    :return: (tuple) centroid x_center, y_center
    """
    # use the bounding box coordinates to derive the centroid
    cX = int((bbox[0] + bbox[2]) / 2.0)
    cY = int((bbox[1] + bbox[3]) / 2.0)

    return cX, cY

