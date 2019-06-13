import os
import numpy as np

from scipy.spatial import distance

from collections import OrderedDict, deque

from scipy.spatial.distance import cdist
from sklearn.utils import linear_assignment_
from utils import kalman_filter
from utils.kalman_filter import KalmanFilter
from models.reid import load_reid_model, extract_reid_features

# ------------------------------------------------------------------------------------------

'''def get_distance_reid(lost_tracks, tracklets):
    dictances = {}
    for index in lost_tracks:
        lost_track = lost_tracks[index]
        for i in tracklets:
            track = tracklets[i]
            dst = distance.euclidean(track, lost_track)
            dictances[i] = np.asarray(dst)'''


"""
Parameters
    ----------
    existing_tracks : existing tracks that need to be matched
    refind_tracks : new tracks with negative ids.
    
Returns
    Matrix of len(existing_tracks) x len(refind_tracks) indicating distance scores.
"""
def nearest_reid_distance(existing_tracks, refind_tracks, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]

    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(existing_tracks), len(refind_tracks)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    refind_track_features = np.asarray([refine_track.curr_feature for refine_track in refind_tracks], dtype=np.float32)
    '''det_features = []
    for i in refind_reid_feature:
        det_features.append(i)'''
    for i, track in enumerate(existing_tracks):
        #track = tracklets_reid_feature[i]
        #cost_matrix[i, :] = distance.euclidean(track._reid_feature, refind_track_features)
        cost_matrix[i, :] = np.maximum(0.0, cdist(track._reid_feature, refind_track_features, metric).min(axis=0))







    return cost_matrix
#=================================
def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix
#=============
def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    indices = linear_assignment_.linear_assignment(cost_matrix)

    return _indices_to_matches(cost_matrix, indices, thresh)
#==============================================================='''
# reimplement tracking stuffs

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def additional_info(det_data):
    s = ''
    for id in range(6, len(det_data)):
        s += ' ' + str(det_data[id])
    return s

def type_str_to_int(str_type) :
    if str_type == 'face' :
        return 1
    elif str_type == 'pedestrian' :
        return 2
    elif str_type == 'vehicle' :
        return 3
    else :
        print('Invalid type : ' + str_type)

class Tracklet :
    def __init__(self) :
        self._object_type = 0 # 'face' = 1, 'ped' = 2 or 'vehicle' > 2 or 'unknown' = 0
        # to indicate the validity of tracklet
        self._valid = True
        # id
        self._id = -1
        self._bounding_box = np.zeros(shape=(4), dtype=np.float32)
        # a property that is being added to better track in occulted areas. The flow will be computed between
        # 2 frames which are far apart, and id will be reconciliated.
        self._bounding_box_in_the_past = {}
        self._speed = np.zeros(shape=(4), dtype=np.float32)
        self._rate_width = 1.0
        self._rate_height = 1.0
        # a buffer of speed to memorize the past
        self._set_of_old_speed = []
        self._buffer_size = 1
        self._last_speed_from_optical_flow = np.zeros(shape=(4), dtype=np.float32)
        # last frame that received an update from detection
        self._old_frame_updated = -1
        # number of frames that have been merged to the tracklet
        self._count_frames = 0
        # other information from det file (bio qual, type, det confidence, etc.)
        self._additional_det_info = ''
        # this dictionary stores the correspondences between new tracklets id and
        # the tracklets id from the baseline result
        self._associated_baseline = -1

        # parameters to deliver
        self._initializing_frames = 3
        self._threshold_on_standby_frames = 25
        self._threshold_on_standby_frames_to_view = 4
        self._threshold_on_standby_frames_to_save = 4

        self._mask = []
        #self._color_vis = () # only to show in the output image
        self._color_vis = np.random.choice(range(256), size=3)

        self._reid_feature = None
        self.curr_feature = None
        self.last_feature = None

    def __init__(self,item, det_or_ref=None, frame_id=None, tracker=None, max_n_features = 1000) :
        if det_or_ref == 'det' :
            int_type = int(item[7])
            self._object_type = int_type
            self._id = -1
            self._bounding_box = np.array([item[2], item[3],
                                  item[2] + item[4], item[3]+item[5]])
            # other information from det file (bio qual, type, det confidence, etc.)
            self._additional_det_info = additional_info(item)
            self._associated_baseline = -1

            # parameters to deliver
            self._initializing_frames = tracker._initializing_frames
            self._threshold_on_standby_frames = tracker._threshold_on_standby_frames
            self._threshold_on_standby_frames_to_view = tracker._threshold_on_standby_frames_to_view
            self._threshold_on_standby_frames_to_save = tracker._threshold_on_standby_frames_to_save

            self._reid_feature = None
            self.max_n_features = max_n_features
            self.curr_feature = None
            self.last_feature = None

            self._reid_feature = deque([], maxlen=self.max_n_features)

        elif det_or_ref == 'ref' :
            object_type = item['type']
            self._object_type = type_str_to_int(object_type)
            self._id = item['id']
            self._bounding_box = np.array([item['x'], item['y'],
                                           item['x'] + item['width'],
                                           item['y'] + item['height']])
            # other information from det file (bio qual, type, det confidence, etc.)
            self._additional_det_info = ''
        else :
            print('det_or_ref argument is not valid : ' + str(det_or_ref))
            exit(1)

        self._valid = True
        self._bounding_box_in_the_past = {}
        self._speed = np.zeros(shape=(4), dtype=np.float32)
        self._rate_width = 1.0
        self._rate_height = 1.0
        # a buffer of speed to memorize the past
        self._set_of_old_speed = []
        self._buffer_size = tracker._buffer_size
        self._last_speed_from_optical_flow = np.zeros(shape=(4), dtype=np.float32)
        # last frame that received an update from detection
        self._old_frame_updated = frame_id
        # number of frames that have been merged to the tracklet
        self._count_frames = 1
        self._mask = None
        self._color_vis = np.random.choice(range(256), size=3)

    def assign_valid_id(self, tracker) :
        self._id = tracker._NEXT_ID
        tracker._NEXT_ID += 1

    def ready_for_new_id(self) :
        return self._id < 0 and self._count_frames >= self._initializing_frames

    # return True if the track is to be kept in memory, False if the track is obsolete and needs to be removed
    def review_track(self, frame_id) :
        if self._valid is False :
            return False
        else :
            if frame_id - self._old_frame_updated >= self._threshold_on_standby_frames :
                self._valid = False
                return False
            else :
                return True

    # save position in the past in order to better track with optical flow
    def save_position(self, re_flow) :
        self._bounding_box_in_the_past[re_flow] = self._bounding_box

    # to decide if a tracklet of id ident is valid to be showed on image
    def valid_to_show(self, frame_id) :
        return self._valid and \
               self._id >= 0 and \
               frame_id - self._old_frame_updated < self._threshold_on_standby_frames_to_view

    # to decide if a tracklet of id ident is valid to be saved in the output file
    def valid_to_save(self, frame_id):
        return self._valid and \
               self._id >= 0 and \
               frame_id - self._old_frame_updated < self._threshold_on_standby_frames_to_save


    def set_feature(self, feature):
        if feature is None:
            return False
        self._reid_feature.append(feature)
        self.curr_feature = feature
        self.last_feature = feature
        # self._p_feature = 0
        return True

