from metric import Metric


def soft_nms(bboxes, threshold=0.75):
    """
    soft non-max suppression implementation for object detection. takes a list of bounding boxes with scores,
    decreases score of bounding boxes with respect to their IoU with the highest score bounding box
    :param bboxes: (array) list of bounding boxes in format [x1, y1, x2, y2, score]
    :param threshold: (scalar, float) minimum acceptable score for bounding box
    :return: filtered list of bounding boxes after soft non-max suppression
    """
    bboxes_nms = []

    while bboxes:
        # filter all the bboxes with score below the threshold
        bboxes = [bbox for bbox in bboxes if bbox[4] >= threshold]

        # grab the highest score bbox, remove it from the list and add it to output list
        maxscore_bbox = max(bboxes, key=lambda x: x[4])
        bboxes.remove(maxscore_bbox)
        bboxes_nms.append(maxscore_bbox)

        # loop over all the possible overlaps with the selected bbox and update their scores
        bbox_pairs = [(maxscore_bbox, bbox) for bbox in bboxes]
        for maxscore_bbox, bbox2 in bbox_pairs:
            bbox2[4] -= Metric.iou(maxscore_bbox, bbox2)

    return bboxes_nms

# print soft_nms([[2, 5, 3, 6, 0.9], [5, 10, 15, 20, 0.98], [80, 210, 100, 230, 0.6], [3, 6, 4, 7, 0.99]])

