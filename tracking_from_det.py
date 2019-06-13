import os
import numpy as np
from numba import jit
import cv2, math
from random import randint
from scipy.cluster.vq import kmeans, vq
from tracking_from_ref_annotation import Tracker_from_ref_annotation
from tracklet import Tracklet, get_iou, additional_info, nearest_reid_distance, gate_cost_matrix, linear_assignment
from show_flo import readFlow

from collections import OrderedDict, deque


from utils.kalman_filter import KalmanFilter
from models.reid import load_reid_model, extract_reid_features


def inside(p, top, left, h, w):
    return p[0] >= top and p[0] < top + h \
           and p[1] >= left and p[1] < left + w


def average_motion_experimental_backup_only(flow, track, width, height, frame_id,
                                method='clustering', padding_out=0.25, padding_in=0.33):
    bounding_box = [int(track._bounding_box[1]), int(track._bounding_box[0]),
                    int(track._bounding_box[3]), int(track._bounding_box[2])]

    # compute out padded bounding box with padding_out
    padding_px_h_out = int(height * padding_out)
    padding_px_w_out = int(width * padding_out)
    padded_out_bb = [bounding_box[0] - padding_px_h_out, bounding_box[1] - padding_px_w_out,
                     bounding_box[2] + padding_px_h_out, bounding_box[3] + padding_px_w_out]
    # cropped if outside image
    padded_out_bb = [max(padded_out_bb[0], 0), max(padded_out_bb[1], 0),
                     min(padded_out_bb[2], flow.shape[0]), min(padded_out_bb[3], flow.shape[1])]
    # to be able to return position relative to the original bounding_box
    top = bounding_box[0] - padded_out_bb[0]
    left = bounding_box[1] - padded_out_bb[1]

    # compute in padded bounding box with padding_in
    padding_px_h_in = int(height * padding_in)
    padding_px_w_in = int(width * padding_in)
    padded_in_bb = [bounding_box[0] + padding_px_h_in, bounding_box[1] + padding_px_w_in,
                    bounding_box[2] - padding_px_h_in, bounding_box[3] - padding_px_w_in]
    # padded_in_bb relative to padded_out_bb
    top_to_in = padded_in_bb[0] - padded_out_bb[0]
    left_to_in = padded_in_bb[1] - padded_out_bb[1]
    height_in = padded_in_bb[2] - padded_in_bb[0]
    width_in = padded_in_bb[3] - padded_in_bb[1]

    all_flow = flow[padded_out_bb[0]:padded_out_bb[2],
               padded_out_bb[1]:padded_out_bb[3], :]
    if all_flow.shape[0] > 0 and all_flow.shape[1] > 0:
        # number of bins and of groups depend on type
        if track._object_type == 1 or track._object_type > 2:  # face or vehicle
            number_of_bins = 2
            number_of_groups_selected = 2
        elif track._object_type == 2:  # pedestrian
            number_of_bins = 3
            number_of_groups_selected = 3
        if method == 'histogram':
            u = all_flow[:, :, 0]
            v = all_flow[:, :, 1]
            hist_uv = np.histogram2d(u.flatten(), v.flatten(), bins=number_of_bins)
            if False:
                if track._id > 0:
                    hist_f = open('out_auxiliary/' + str(track._id) + '_' + str(frame_id) + '.txt', 'w')
                    hist_f.write(str(hist_uv[0]) + '\n')
                    hist_f.write(str(hist_uv[1]) + '\n')
                    hist_f.write(str(hist_uv[2]) + '\n')
                    hist_f.close()
            min_dist = 0
            last_set = None
            if False:
                # a map to visualize all groups of points
                map_to_save = np.ones(shape=(int(height), int(width), 3), dtype=np.uint8)
            for i in range(number_of_groups_selected):
                id_max = hist_uv[0].argmax()
                id_max_x = int(id_max / hist_uv[0].shape[1])
                id_max_y = id_max % hist_uv[0].shape[1]
                lower_bound_u = hist_uv[1][id_max_x]
                upper_bound_u = hist_uv[1][id_max_x + 1]
                lower_bound_v = hist_uv[2][id_max_y]
                upper_bound_v = hist_uv[2][id_max_y + 1]
                hist_uv[0][id_max_x, id_max_y] = -1

                set = [[x, y] for x in range(all_flow.shape[0]) for y in range(all_flow.shape[1]) \
                       if all_flow[x, y, 0] >= lower_bound_u and all_flow[x, y, 0] <= upper_bound_u \
                       and all_flow[x, y, 1] >= lower_bound_v and all_flow[x, y, 1] <= upper_bound_v]
                dist_set = [abs(v[1] - width / 2.0) for v in set]
                dist_set = sum(dist_set) / len(dist_set)
                if last_set is None or dist_set < min_dist:
                    last_set = set
                    min_dist = dist_set

                # modify map
                random_color = np.random.choice(range(256), size=3)
                if False:
                    for p in set:
                        try:
                            map_to_save[p[0], p[1]] = random_color
                        except:
                            print('except')

            # update mask with last_set
            track._mask = last_set

            all_valid_values = [all_flow[v[0], v[1]] for v in track._mask]
            all_valid_values = sum(all_valid_values) / len(all_valid_values)

            # for the real mask, colorize in red
            if False:
                for p in track._mask:
                    try:
                        map_to_save[p[0], p[1]] = (0, 0, 255)
                    except:
                        print('except')
                cv2.imwrite('out_auxiliary/' + str(track._id) + '_' + str(frame_id) + '.png', map_to_save)
        elif method == 'clustering':
            data = np.reshape(all_flow, (-1, 2))
            centroids, _ = kmeans(data, number_of_bins)
            idx, _ = vq(data, centroids)
            all_sets = []  # set of potential masks
            outsiders = []  # counter of points inside padded_bounding_box but
            insiders = []  # counter of points in each set
            # outsize bounding_box
            for _ in range(number_of_bins):
                all_sets.append([])
                outsiders.append(0)
                insiders.append(0)
            for row in range(idx.shape[0]):
                id_x = int(row / all_flow.shape[1])
                id_y = row % all_flow.shape[1]
                to_add = True
                if to_add:
                    if inside([id_x, id_y], top, left, height, width):
                        all_sets[idx[row]].append([id_x - top, id_y - left])
                    else:
                        outsiders[idx[row]] += 1
                    if inside([id_x, id_y], top_to_in, left_to_in, height_in, width_in):
                        insiders[idx[row]] += 1

            if True:
                for i in range(number_of_bins):
                    # normalize
                    if insiders[i] > 0:
                        outsiders[i] /= insiders[i]
                    else:
                        outsiders[i] = 1000.0  # max value

            min_dist = 1000.0
            last_set = None
            c = 0
            last_c = None
            for id_set in range(len(all_sets)):
                set = all_sets[id_set]
                dist_set = outsiders[id_set]
                if last_set is None or dist_set < min_dist:
                    last_set = set
                    min_dist = dist_set
                    last_c = c

                c += 1

            # update mask with last_set
            track._mask = last_set
            # check validity
            all_valid_values = centroids[last_c]
        return all_valid_values, True
    else:
        return None, False

class Tracker_from_det(Tracker_from_ref_annotation) :
    def __init__(self, threshold_iou = 0.5, max_n_features =1000, min_ap_dist=0.64, initializing_frames=3, threshold_on_standby_frames=25, threshold_on_standby_frames_to_view=4, threshold_on_standby_frames_to_save=4,
                 buffer_size=1, # buffer size of speed for each tracklet created
                 ) :
        super(Tracker_from_det, self).__init__(threshold_iou, initializing_frames, threshold_on_standby_frames, threshold_on_standby_frames_to_view, threshold_on_standby_frames_to_save,
                 buffer_size)

        self._reid_feature = None
        self.kalman_filter = None

        self.reid_model = load_reid_model()
        self.kalman_filter = KalmanFilter()
        self.max_n_features = max_n_features
        self.min_ap_dist = min_ap_dist
        self.curr_feature = None
        self.last_feature = None

        # ===============================================================================================

    '''def matching_reid_replace_iou(self, dets_data):
        if dets_data is not None:
            for index in self._tracklets :
                track = self._tracklets[index]
                if track._valid :
                    track_box = track._bounding_box
                    for i in range(len(dets_data)) :
                        int_type_det = int(dets_data[i][7])
                        if int_type_det == track._object_type or \
                                (int_type_det > 2 and track._object_type > 2) : # we track modality by modality
                            det_box = [dets_data[i][2], dets_data[i][3],
                                       dets_data[i][2] + dets_data[i][4],
                                       dets_data[i][3] + dets_data[i][5]]'''






    def matching_score_iou(self, dets_data) :
        matching_scores = []
        #lost_track = []
        if dets_data is not None:
            for index in self._tracklets :
                track = self._tracklets[index]
                if track._valid :
                    track_box = track._bounding_box  # left top right bottom
                    for i in range(len(dets_data)) :
                        int_type_det = int(dets_data[i][7])
                        if int_type_det == track._object_type or \
                                (int_type_det > 2 and track._object_type > 2) : # we track modality by modality
                            det_box = [dets_data[i][2], dets_data[i][3],
                                       dets_data[i][2] + dets_data[i][4],
                                       dets_data[i][3] + dets_data[i][5]]
                            # compute iou
                            score = get_iou({'x1': track_box[0], 'y1': track_box[1], 'x2': track_box[2], 'y2': track_box[3]},
                                            {'x1': det_box[0], 'y1': det_box[1], 'x2': det_box[2], 'y2': det_box[3]})
                        else :
                            score = 0.0
                        if score > self._threshold_iou :
                            matching_scores.append(
                                (index, i, score))  # identity of track, id of det in the list of dets, score

        return matching_scores

    def extract_reid_abandonne_track(self, set_of_abandonned_index, image):
        bdb_tracks =[self._tracklets[id]._bounding_box for id in set_of_abandonned_index]

        features = extract_reid_features(self.reid_model, image, bdb_tracks)  # dung CNN de extract cls_feature to
        features = features.cpu().numpy()

        for i, index in enumerate(set_of_abandonned_index):
            track = self._tracklets[index]
            track.set_feature(features[i])

            # track._bounding_box.set_feature(features[id_track])'''




    '''def extract_reid_feature(self, image, ids_extract_feature):
        bdb_tracks = []
        #for id_track in ids_of_updated_tracks:
        for id_track in ids_of_extract_feature:
            track = self._tracklets[id_track]
            bdb_tracks.append(track._bounding_box)
            # set feature Re_ID
        features = extract_reid_features(self.reid_model, image, bdb_tracks )  # dung CNN de extract cls_feature to
        features = features.cpu().numpy()
        #count = 0
        for id_track in ids_extract_feature:
            track = self._tracklets[id_track]
            track.set_feature(features[id_track])
            track._reid_feature[id_track] = features[id_track]
            #count += 1
            #track._bounding_box.set_feature(features[id_track])'''



    def refine_matching_score(self, matching_scores, dets_data) :
        new_matching_scores = []
        abandonned_tracks = []
        new_dets = []
        accepted_tracks = []
        scores_by_det = {}
        for m in matching_scores:
            # for each det, get the track with highest iou score
            if m[1] not in scores_by_det or m[2] > scores_by_det[m[1]][2]:
                scores_by_det[m[1]] = m
        for i in scores_by_det :
            new_matching_scores.append(scores_by_det[i])
            accepted_tracks.append(scores_by_det[i][0])
        for index in self._tracklets :
            if index not in accepted_tracks:
                abandonned_tracks.append(index)
        for i in range(len(dets_data)) :
            if i not in scores_by_det :
                new_dets.append(i)

        return new_matching_scores, abandonned_tracks, new_dets


    def update_tracks(self, dets_data, matching_scores, frame_id, image) :
        for m in matching_scores :


            new_bounding_box = np.array([dets_data[m[1]][2], dets_data[m[1]][3],
                                                   dets_data[m[1]][2] + dets_data[m[1]][4],
                                                   dets_data[m[1]][3] + dets_data[m[1]][5]])
            new_bounding_box = 0.5*self._tracklets[m[0]]._bounding_box + 0.5*new_bounding_box
            self._tracklets[m[0]]._bounding_box =  new_bounding_box
            self._tracklets[m[0]]._old_frame_updated = frame_id
            self._tracklets[m[0]]._additional_det_info = additional_info(dets_data[m[1]])
            self._tracklets[m[0]]._count_frames += 1
        bdb_tracks = []
        for m in matching_scores:
            track = self._tracklets[m[0]]
            bdb_tracks.append(track._bounding_box)
            # set feature Re_ID
        features = extract_reid_features(self.reid_model, image, bdb_tracks)  # dung CNN de extract cls_feature to
        features = features.cpu().numpy()

        count = 0
        for m in matching_scores:
            track = self._tracklets[m[0]]
            track.set_feature(features[count])
            count +=1


            '''for m in matches:
                new_bounding_box = self._tracklets[m[0]]._bounding_box
                self._tracklets[m[0]]._bounding_box = new_bounding_box'''

            # set feature Re_ID



            # check if we need a new id
            #if self._tracklets[m[0]].ready_for_new_id() :
            #    self._tracklets[m[0]].assign_valid_id(tracker=self)

    def update_tracks_non_reid(self, dets_data, matching_scores, frame_id):
        for m in matching_scores:
            new_bounding_box = np.array([dets_data[m[1]][2], dets_data[m[1]][3],
                                         dets_data[m[1]][2] + dets_data[m[1]][4],
                                         dets_data[m[1]][3] + dets_data[m[1]][5]])
            new_bounding_box = 0.5 * self._tracklets[m[0]]._bounding_box + 0.5 * new_bounding_box
            self._tracklets[m[0]]._bounding_box = new_bounding_box
            self._tracklets[m[0]]._old_frame_updated = frame_id
            self._tracklets[m[0]]._additional_det_info = additional_info(dets_data[m[1]])
            self._tracklets[m[0]]._count_frames += 1
            # check if we need a new id
            if self._tracklets[m[0]].ready_for_new_id():
                self._tracklets[m[0]].assign_valid_id(tracker=self)
    def remove_abandonned_tracks(self, abandonned_tracks) :
        # first, count the number of tracks without id that will be removed
        removed_tracks_noid = [self._tracklets[index] for index in abandonned_tracks
                               if self._tracklets[index]._id == -1]
        self._number_tracks_noid += len(removed_tracks_noid)
        for id in abandonned_tracks :
            self._tracklets.pop(id)

    def add_new_tracks_for_first_frame(self,dets_data, new_dets, frame_id):
        self._current_boxes_baseline.clear()
        for i in new_dets:
            new_det = dets_data[i]
            new_track = Tracklet(item=new_det, det_or_ref='det', frame_id=frame_id, tracker=self)
            # check if we need a new id
            if new_track.ready_for_new_id() :
                new_track.assign_valid_id(tracker=self)
            if len(self._tracklets) == 0:
                next_index = 0
            else:
                next_index = max(self._tracklets.keys()) + 1
            self._tracklets[next_index] = new_track

    def add_new_tracks(self, dets_data, new_dets, frame_id, image) :
        self._current_boxes_baseline.clear()
        bdb_tracks = []
        index_added_to_tracker = []
        for i in new_dets :
            new_det = dets_data[i]
            new_track = Tracklet(item=new_det, det_or_ref='det', frame_id=frame_id, tracker=self)
            # check if we need a new id
            #if new_track.ready_for_new_id() :
            #    new_track.assign_valid_id(tracker=self)
            if len(self._tracklets) == 0 :
                next_index = 0
            else :
                next_index = max(self._tracklets.keys()) + 1
            self._tracklets[next_index] = new_track
            index_added_to_tracker.append(next_index)

        # feature reid
            bdb_tracks.append(new_track._bounding_box)
            # set feature Re_ID
        features = extract_reid_features(self.reid_model, image, bdb_tracks)  # dung CNN de extract cls_feature to
        features = features.cpu().numpy()
        for i in range(len(bdb_tracks)):
            tracklet = self._tracklets[index_added_to_tracker[i]]
            tracklet.set_feature(features[i])
    # ===========================================================================
    def matching_reid_backup(self, ids_tracks_to_matching_reid):
        tracklets_reid_feature = {}
        for index in self._tracklets:
            #if self._tracklets[index]._id > 0:
            track = self._tracklets[index]
            tracklets_reid_feature[index] = track

        matches = []

        for m in ids_tracks_to_matching_reid:
            refind_reid_feature = []
            if self._tracklets[m]._id < 0:
                refind_reid_feature.append(self._tracklets[m])
                dists = nearest_reid_distance(tracklets_reid_feature, refind_reid_feature, metric='euclidean')
                matche = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
                matche = list(matche)
                matches.append(matche)
                #dists = gate_cost_matrix(self.kalman_filter, dists, self._tracklets[m[0]], refind_reid_feature)
                #matches, u_track, u_detection = linear_assignment(dists, thresh=self.min_ap_dist)
        return matches
#-----------------------------------------------------------------------------------------------------
    def matching_reid(self, set_of_negative_index=[], set_of_abandonned_index=[]):

        matches = []
        dists = nearest_reid_distance(existing_tracks=[self._tracklets[id] for id in set_of_abandonned_index],
                                      refind_tracks=[self._tracklets[id] for id in set_of_negative_index],
                                      metric='euclidean')

        if dists.shape[0] * dists.shape[1] > 0 :
            dists_argmin = np.argmin(dists, axis=1)
            dists_min = [dists[i, dists_argmin[i]] for i in range(len(dists_argmin))]
            for i in range(len(dists_argmin)) :
                if dists_min[i] < 0.64 :
                    matches.append([set_of_abandonned_index[i], set_of_negative_index[dists_argmin[i]]])

        #matche = list(matche)
        #matches.append(match)
        #for i in range(len(dists)) :
            '''j = int(dists[i][0])
            matches.append([set_of_abandonned_index[i], set_of_negative_index[j]])
                #dists = gate_cost_matrix(self.kalman_filter, dists, self._tracklets[m[0]], refind_reid_feature)
                #matches, u_track, u_detection = linear_assignment(dists, thresh=self.min_ap_dist)'''
        return matches

    # --------------------------------------------------------------------------------------
    def update_tracks_reid(self, matche, frame_id):
        for index_abandonned, index_refind_track in matche :
            #new_bounding_box = self._tracklets[index_refind_track]._bounding_box
            #self._tracklets[index_abandonned]._bounding_box = new_bounding_box
            #self._tracklets[index_abandonned]._old_frame_updated = frame_id
            #self._tracklets[id_track]._additional_det_info = additional_info(dets_data[m[1]])
            #self._tracklets[index_abandonned]._count_frames += 1
            self._tracklets[index_refind_track]._id = self._tracklets[index_abandonned]._id


        self.remove_abandonned_tracks([t[0] for t in matche])



    def ready_for_new_id(self, matches):
        for m in matches:
            if self._tracklets[m[0]].ready_for_new_id():
                self._tracklets[m[0]].assign_valid_id(tracker=self)




    #-------------------------------------------------------------------------------------------
    def review_all_tracks(self, frame_id) :
        removed = []
        for index in self._tracklets :
            track = self._tracklets[index]
            if track.review_track(frame_id) is False :
                removed.append(index)

        self.remove_abandonned_tracks(removed)

    def combine_forward_backward(self, flow, flow_bw) :
        w = flow.shape[1]
        h = flow.shape[0]
        for id_w in range(w) :
            for id_h in range(h) :
                backward_f = flow_bw[id_h, id_w, :]
                new_id_w = int(id_w + backward_f[0])
                new_id_h = int(id_h + backward_f[1])
                # check if this new point is still inside image
                if new_id_w >= 0 and new_id_w < w and new_id_h >= 0 and new_id_h < h :
                    forward_f = flow[new_id_h, new_id_w, :]
                    # reverse the backward flow
                    backward_f = - backward_f
                    # get the average between forward and backward flow
                    forward_f = (forward_f + backward_f) / 2.0
                    # update flow
                    flow[new_id_h, new_id_w, :] = forward_f
        return flow

    def save_position(self, re_flow) :
        for index in self._tracklets:
            track = self._tracklets[index]
            if track._valid:
                track.save_position(re_flow)

    def estimate_track_speed_from_flow_backup_only(self, frame_id, re_flow, flo_dir, threshold_uncertainty=0) :
        if frame_id%re_flow == 0 :
            # load optical flow
            if flo_dir is not None :
                flo_fn = flo_dir + '/frame' + str(frame_id) + '.png_flow[0].fwd.flo'
                if not os.path.exists(flo_fn):
                    flo_fn = flo_dir + '/frame' + str(frame_id) + '.png.flo'
                    if not os.path.exists(flo_fn) :
                        flo_fn = flo_dir + '/frame' + str(frame_id) + '.flo'
                        if not os.path.exists(flo_fn) :
                            print(flo_fn + ' does not exist')
                            return
            else :
                flo_fn = None
        else :
            flo_fn = None

        # TODO update current_speed
        if flo_fn is not None and os.path.exists(flo_fn) :
            flow = readFlow(flo_fn)
        elif flo_fn is not None :
            print(flo_fn + ' does not exist')
            exit(1)
        else :
            flow = None

        # get global speed for all point
        removed = []
        adapt_size = False
        number_sampling = 20
        for id_track in self._tracklets :
            track = self._tracklets[id_track]
            if track._valid :
                if frame_id%re_flow == 0 :
                    # first case : at the beginning of the flow cycle
                    if np.sum(np.abs(track._last_speed_from_optical_flow)) > 0.0 and frame_id - track._old_frame_updated < threshold_uncertainty :
                        # if we are confidence enough, add the last speed to old speed
                        track._set_of_old_speed.append(track._last_speed_from_optical_flow)
                        track._set_of_old_speed = track._set_of_old_speed[-track._buffer_size:]

                    if frame_id - track._old_frame_updated < threshold_uncertainty :
                        # we only update speed from optical flow if the track has been updated recently
                        if flow is None :
                            track._last_speed_from_optical_flow = np.zeros(shape=(4), dtype=np.float32)
                            # update speed with the last one added
                            track._speed = track._last_speed_from_optical_flow
                        else :
                            height = track._bounding_box[3] - track._bounding_box[1]
                            width = track._bounding_box[2] - track._bounding_box[0]
                            if width <= 0 or height <= 0:
                                track._valid = False
                                removed.append(id_track)
                                continue

                            #all_flow, valid = average_motion(flow, track, width, height, track._object_type)
                            all_flow, valid = average_motion_experimental(flow, track, width, height, frame_id, 'clustering')
                            if valid == False :
                                track._valid = False
                                removed.append(id_track)
                                continue

                            if all_flow is not None :
                                track._last_speed_from_optical_flow = np.asarray([all_flow[0], all_flow[1], all_flow[0], all_flow[1]], dtype=np.float32)
                                if adapt_size :
                                    if len(track._mask) == 0 :
                                        track._rate_width = 1.0
                                        track._rate_height = 1.0
                                    else :
                                        rate_w = []
                                        rate_h = []
                                        for _ in range(number_sampling) :
                                            r_p1 = randint(0, len(track._mask)-1)
                                            r_p2 = randint(0, len(track._mask)-1)
                                            if r_p1 == r_p2 :
                                                continue
                                            r_p1 = track._mask[r_p1]
                                            r_p1 = [track._bounding_box[1]+r_p1[0], track._bounding_box[0]+r_p1[1]]
                                            v1 = flow[int(r_p1[0]), int(r_p1[1]), :] / re_flow
                                            r_p2 = track._mask[r_p2]
                                            r_p2 = [track._bounding_box[1] + r_p2[0], track._bounding_box[0] + r_p2[1]]
                                            v2 = flow[int(r_p2[0]), int(r_p2[1]), :] / re_flow
                                            old_h = abs(r_p2[0] - r_p1[0])
                                            new_h = abs(r_p2[0] + v2[1] - r_p1[0] - v1[1])
                                            old_w = abs(r_p2[1] - r_p1[1])
                                            new_w = abs(r_p2[1] + v2[0] - r_p1[1] - v1[0])
                                            if old_h > height/4 and old_w > width/4 :
                                                rate_w.append(new_w / old_w)
                                                rate_h.append(new_h / old_h)

                                        if len(rate_w) > 0 :
                                            track._rate_width = sum(rate_w) / len(rate_w)
                                            track._rate_height = sum(rate_h) / len(rate_h)
                                        else :
                                            track._rate_width = 1.0
                                            track._rate_height = 1.0
                                            print('No sampling')

                                # the flow has been computed for re_flow frames, so the exact speed is obtained
                                # by dividing these quantities by re_flow
                                track._last_speed_from_optical_flow /= re_flow
                                # update speed with the last one added
                                track._speed = track._last_speed_from_optical_flow
                            else :
                                # in case of occultation, the speed is the average of all speed in (unchanged) _set_of_speed
                                track._speed = np.zeros(shape=(4), dtype=np.float32)
                                for s in track._set_of_old_speed:
                                    track._speed += s
                                if len(track._set_of_old_speed) > 0:
                                    track._speed /= len(track._set_of_old_speed)
                                    track._speed *= 1.1
                    else :
                        # in case of occultation, the speed is the average of all speed in (unchanged) _set_of_speed
                        track._speed = np.zeros(shape=(4), dtype=np.float32)
                        for s in track._set_of_old_speed :
                            track._speed += s
                        if len(track._set_of_old_speed) > 0 :
                            track._speed /= len(track._set_of_old_speed)
                            track._speed *= 1.1

                else :
                    # second case, we are in the middle of the flow cycle
                    if frame_id - track._old_frame_updated < threshold_uncertainty :
                        # if we are confident enough, continue to use _last_speed_from_optical_flow
                        track._speed = track._last_speed_from_optical_flow
                    else :
                        # in case of occultation, the speed is the average of all speed in (unchanged) _set_of_speed
                        track._speed = np.zeros(shape=(4), dtype=np.float32)
                        for s in track._set_of_old_speed:
                            track._speed += s
                        if len(track._set_of_old_speed) > 0:
                            track._speed /= len(track._set_of_old_speed)
                            track._speed *= 1.1

        self.remove_abandonned_tracks(removed)

    def show_on_image_backup_only(self, image, frame_id, re_det, show_mask=False, show_arrow=False) :
        for id_track in range(len(self._tracklets)) :
            track = self._tracklets[id_track]
            if track.valid_to_show(frame_id) :
                color_rectangle = track._color_vis.tolist()
                if frame_id - track._old_frame_updated >= re_det :
                    color_rectangle = (0, 255, 0)
                if show_mask and len(track._mask) > 0 :
                    for p in track._mask :
                        try :
                            image[int(track._bounding_box[1])+p[0], int(track._bounding_box[0])+p[1]] = track._color_vis
                        except :
                            print('Except')
                cv2.rectangle(image, (int(track._bounding_box[0]),
                                      int(track._bounding_box[1])),
                              (int(track._bounding_box[2]),
                               int(track._bounding_box[3])), color_rectangle,
                              thickness=2)
                cv2.putText(image, str(track._id),
                            (int(track._bounding_box[0]) + 2, int(track._bounding_box[1]) + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 0, 0), thickness=2)
                if show_arrow and len(track._mask) > 0 :
                    centre = [0, 0]
                    for p in track._mask :
                        centre[0] += p[0]
                        centre[1] += p[1]
                    centre[0] = centre[0] / len(track._mask) + track._bounding_box[1]
                    centre[1] = centre[1] / len(track._mask) + track._bounding_box[0]
                    end_p = (int(centre[1] + track._speed[0]*5), int(centre[0] + track._speed[1]*5))
                    begin_p = (int(centre[1]), int(centre[0]))
                    cv2.arrowedLine(image, begin_p, end_p, color=(255, 0, 0), thickness=2)

    # overriding save_to_txt function
    def save_to_txt(self, out_file, frame_id) :
        for index in self._tracklets :
            track = self._tracklets[index]
            if track.valid_to_save(frame_id) :
                s = str(frame_id) + ' ' + str(track._id) + ' ' + str(int(track._bounding_box[0]))\
                + ' ' + str(int(track._bounding_box[1])) + ' ' + str(int(track._bounding_box[2] - track._bounding_box[0]))\
                + ' ' + str(int(track._bounding_box[3] - track._bounding_box[1])) + track._additional_det_info
                out_file.write(s + '\n')

    def no_valid_track(self) :
        return all(not self._tracklets[index]._valid for index in self._tracklets)

    def print_info(self, filename=None) :
        if filename is not None :
            f_ = open(filename, 'w')
            f_.write('threshold_iou : ' + str(self._threshold_iou) + '\n')
            f_.write('Found : ' + str(self._NEXT_ID) + ' tracklets\n')
            f_.write('NEXT_ID : ' + str(self._NEXT_ID) + '\n')
            f_.write('Single detection tracks : ' + str(self._number_tracks_noid) + '\n')
            f_.close()
        else :
            print('threshold_iou : ' + str(self._threshold_iou))
            print('Found : ' + str(self._NEXT_ID) + ' tracklets')
            print('NEXT_ID : ' + str(self._NEXT_ID))
            print('Single detection tracks : ' + str(self._number_tracks_noid))

# ------------------------------------------------------------------------------------------