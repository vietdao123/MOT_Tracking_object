import os
import cv2
import math
import numpy as np
from tracklet import Tracklet, get_iou
from scipy.cluster.vq import kmeans, vq
from show_flo import readFlow

def inside(p, top, left, h, w):
    return p[0] >= top and p[0] < top + h \
           and p[1] >= left and p[1] < left + w

def average_motion(flow, track, width, height, object_type):
    if object_type == 1 or object_type > 2:  # faces or vehicles
        all_flow = flow[int(track._bounding_box[1] + height / 4.0):int(
            track._bounding_box[3] - height / 4.0),
                   int(track._bounding_box[0] + width / 4.0):int(
                       track._bounding_box[2] - width / 4.0), :]
    elif object_type == 2:  # pedestrians
        all_flow = flow[int(track._bounding_box[1] + height / 4.0):int(
            track._bounding_box[3] - height / 4.0),
                   int(track._bounding_box[0] + width / 2.0):int(
                       track._bounding_box[0] + width / 2.0 + 1), :]
    else:
        all_flow = None

    if all_flow.shape[0] > 0 and all_flow.shape[1] > 0:
        all_flow = np.mean(all_flow, axis=(0, 1))
        return all_flow, True
    else:
        return None, False

def estimate_mask(flow, track, width, height, frame_id,
                                method='clustering', padding_out=0.25, padding_in=0.33):
    # reset mask
    track._mask = np.zeros(shape=[flow.shape[0], flow.shape[1]], dtype=np.ubyte)

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
            number_of_bins = 3
            number_of_groups_selected = 3
        elif track._object_type == 2:  # pedestrian
            number_of_bins = 3
            number_of_groups_selected = 3
        if method == 'clustering':
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
            for p in last_set :
                orig_p = [int(track._bounding_box[1]+p[0]), int(track._bounding_box[0]+p[1])]
                if orig_p[0] >= 0 and orig_p[0] < flow.shape[0] and orig_p[1] >= 0\
                    and orig_p[1] < flow.shape[1] :
                    track._mask[orig_p[0], orig_p[1]] = np.ubyte(1)
            # check validity
            all_valid_values = centroids[last_c]
        return all_valid_values, True
    else:
        return None, False

class Tracker_from_ref_annotation() :
    def __init__(self, threshold_iou = 0.25, initializing_frames=3, threshold_on_standby_frames=25, threshold_on_standby_frames_to_view=4, threshold_on_standby_frames_to_save=4,
                 buffer_size=1):
        self._threshold_iou = threshold_iou
        self._initializing_frames = initializing_frames
        self._threshold_on_standby_frames = threshold_on_standby_frames
        self._threshold_on_standby_frames_to_view = threshold_on_standby_frames_to_view
        self._threshold_on_standby_frames_to_save = threshold_on_standby_frames_to_save
        self._NEXT_ID = 0
        # to ease the comparison with reference, self._tracklets is reorganized as a dictionary
        self._tracklets = {}
        # set of boxes from baseline and not covered by our boxes
        self._current_boxes_baseline = {}
        # count of number of baseline tracklets that are no longer covered by this
        # new version
        self._id_boxes_baseline = []
        self._number_tracks_noid = 0
        self._buffer_size = buffer_size
        # this dict is for storing errors by id and by frame
        self._error_summary = {}
        print('Tracker initialized with threshold_iou = ' + str(self._threshold_iou))
        print('buffer_size : ' + str(buffer_size))

    def empty(self) :
        return len(self._tracklets) == 0

    def add_to_error_summary(self, vatic_annotation=None, frame_id=0) :
        if vatic_annotation['id'] in self._error_summary :
            print('ERROR : ' + str(vatic_annotation['id']) + ' already exists in error summary')
            exit(1)
        to_add = {'first_frame' : frame_id, 'lost' : None}
        self._error_summary[vatic_annotation['id']] = to_add

    def add_new_tracks(self, dets_data, new_dets, frame_id):
        for i in new_dets:
            new_det = dets_data[i]
            if new_det['occluded'] is False and new_det['outside'] is False :
                new_track = Tracklet(item=new_det, det_or_ref='ref', frame_id=frame_id, tracker=self)
                new_track._mask = None
                if new_track._object_type is not None :
                    self.add_to_error_summary(new_det, frame_id)
                    self._tracklets[new_track._id] = new_track

    def review_all_tracks(self, frame_id) :
        # for the reference version, do nothing
        print('DO NOTHING!')

    def remove_abandonned_tracks(self, abandonned_tracks) :
        # for the reference version, do nothing
        print('DO NOTHING!')

    def update_error_summary_with_one_item(self, ref_object, current_object, frame_id, force_same_size=False):
        if ref_object['occluded'] is False and ref_object['outside'] is False :
            ref_bb = np.array([ref_object['x'], ref_object['y'], ref_object['x']+ref_object['width'],
                               ref_object['y']+ref_object['height']])
            current_bb = current_object._bounding_box

            # modify predicted bounding box to have same size with reference
            if force_same_size :
                diff_width = ref_object['width'] - (current_bb[2]-current_bb[0])
                diff_height = ref_object['height'] - (current_bb[3]-current_bb[1])
                current_bb[0] -= diff_width / 2.0
                current_bb[1] -= diff_height / 2.0
                current_bb[2] += diff_width / 2.0
                current_bb[3] += diff_height / 2.0

            # get iou between 2 boxes
            try :
                score_iou = get_iou({'x1': ref_bb[0], 'y1': ref_bb[1], 'x2': ref_bb[2], 'y2': ref_bb[3]},
                                    {'x1': current_bb[0], 'y1': current_bb[1], 'x2': current_bb[2], 'y2': current_bb[3]})
            except :
                score_iou = 0.0

            # get other scores
            diff = current_bb - ref_bb
            # the euclidean distance between top-left and bottom-right
            diff = (math.sqrt((diff[0]**2 + diff[1]**2) / 2.0) + math.sqrt((diff[2]**2 + diff[3]**2) / 2.0)) / 2.0
            diff_relative = diff / math.sqrt(ref_object['width']**2 + ref_object['height']**2)

            # add to error_summary
            self._error_summary[ref_object['id']][str(frame_id)+'_iou'] = score_iou
            if score_iou == 0.0 :
                self._error_summary[ref_object['id']]['lost'] = frame_id
            self._error_summary[ref_object['id']][str(frame_id)+'_px'] = diff
            self._error_summary[ref_object['id']][str(frame_id)+'_relative'] = diff_relative

    # return abandonned_tracks, new_dets
    def update_error_summary(self, array_r_data, frame_id) :
        new_dets = []
        abandonned_tracks = []
        for id_ref_object in range(len(array_r_data)) :
            ref_object = array_r_data[id_ref_object]
            id = ref_object['id']
            if id not in self._tracklets :
                # new object
                new_dets.append(id_ref_object)
            else :
                current_object = self._tracklets[id]
                self.update_error_summary_with_one_item(ref_object, current_object, frame_id)

        return abandonned_tracks, new_dets

    def estimate_track_speed_from_flow(self, frame_id, re_flow, flo_dir, do_estimate_mask=True, adapt_size=True):
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
        for id_track in self._tracklets:
            track = self._tracklets[id_track]
            if track._valid:
                if frame_id % re_flow == 0:
                    # first case : at the beginning of the flow cycle
                    if np.sum(np.abs(
                            track._last_speed_from_optical_flow)) > 0.0 :
                        # if we are confidence enough, add the last speed to old speed
                        track._set_of_old_speed.append(track._last_speed_from_optical_flow)
                        track._set_of_old_speed = track._set_of_old_speed[-track._buffer_size:]

                    if flow is None:
                        track._last_speed_from_optical_flow = np.zeros(shape=(4), dtype=np.float32)
                    else:
                        height = track._bounding_box[3] - track._bounding_box[1]
                        width = track._bounding_box[2] - track._bounding_box[0]
                        if width <= 0 or height <= 0:
                            track._valid = False
                            removed.append(id_track)
                            continue

                        # now, estimate mask
                        if do_estimate_mask :
                            # estimate mask before other processings
                            all_flow, valid = estimate_mask(flow, track, width, height, frame_id, 'clustering')
                        else :
                            # simply get the average motion
                            all_flow, valid = average_motion(flow, track, width, height, track._object_type)
                        if valid == False:
                            track._valid = False
                            removed.append(id_track)
                            continue

                        if not do_estimate_mask :
                            # if not mask, just use the output of the above function to track
                            if all_flow is not None :
                                track._last_speed_from_optical_flow = np.asarray(
                                    [all_flow[0], all_flow[1], all_flow[0], all_flow[1]], dtype=np.float32)
                                # the flow has been computed for re_flow frames, so the exact speed is obtained
                                # by dividing these quantities by re_flow
                                track._last_speed_from_optical_flow /= re_flow
                                track._speed = track._last_speed_from_optical_flow

        self.remove_abandonned_tracks(removed)

        if do_estimate_mask :
            # this passage is for refining the masks of all tracks
            uncertain_tracks = []
            for id_track in self._tracklets :
                track = self._tracklets[id_track]
                if track._valid :
                    sum_track = np.sum(track._mask)
                    if track._mask is None or sum_track == 0 :
                        uncertain_tracks.append(id_track)
                    else :
                        for id_track_j in self._tracklets :
                            if id_track_j > id_track :
                                track_j = self._tracklets[id_track_j]
                                try :
                                    iou = get_iou({'x1': track._bounding_box[0], 'y1': track._bounding_box[1], 'x2': track._bounding_box[2], 'y2': track._bounding_box[3]},
                                                    {'x1': track_j._bounding_box[0], 'y1': track_j._bounding_box[1], 'x2': track_j._bounding_box[2],
                                                     'y2': track_j._bounding_box[3]})
                                except :
                                    iou = 0
                                if track_j._mask is not None and iou > 0 :
                                    new_mask = track._mask
                                    if track._object_type != 1 :
                                        new_mask = track._mask * (1 - track_j._mask)
                                    new_mask_j = track_j._mask
                                    if track_j._object_type != 1 :
                                        new_mask_j = track_j._mask * (1 - track._mask)
                                    track._mask = new_mask
                                    track_j._mask = new_mask_j
                                    if np.sum(track._mask) < 0.1 * (track._bounding_box[2]-track._bounding_box[0])\
                                            * (track._bounding_box[3]-track._bounding_box[1]):
                                        uncertain_tracks.append(id_track)
                                    if np.sum(track_j._mask) < 0.1 * (track_j._bounding_box[2]-track_j._bounding_box[0])\
                                            * (track_j._bounding_box[3]-track_j._bounding_box[1]):
                                        uncertain_tracks.append(id_track_j)

        # after refinement, used refined mask to estimate motion
        if do_estimate_mask :
            for id_track in self._tracklets :
                track = self._tracklets[id_track]
                if track._valid :
                    if id_track not in uncertain_tracks :
                        all_flow = flow[track._mask == 1]
                        # for translation motion, just get median flow
                        all_flow = np.median(all_flow, axis=0)
                        track._last_speed_from_optical_flow = np.asarray(
                            [all_flow[0], all_flow[1], all_flow[0], all_flow[1]], dtype=np.float32)
                        if adapt_size :
                            # estimate changes in size
                            all_mask_p = np.where(track._mask == 1)
                            l = all_mask_p[0].shape[0]
                            all_mask_p = [[all_mask_p[0][id], all_mask_p[1][id]] for id in range(l)]
                            all_mask_p = sorted(all_mask_p, key=lambda t: t[0] + t[1])
                            N = int(l/4)
                            rate_w = []
                            rate_h = []
                            for t in range(N) :
                                p1 = all_mask_p[t]
                                p2 = all_mask_p[l-1-t]
                                p1_flow = flow[p1[0], p1[1]] / re_flow
                                p1_after = [p1[0]+p1_flow[1], p1[1]+p1_flow[0]]
                                p2_flow = flow[p2[0], p2[1]] / re_flow
                                p2_after = [p2[0]+p2_flow[1], p2[1]+p2_flow[0]]
                                try :
                                    if p2[0] != p1[0] :
                                        rate_h.append((p2_after[0]-p1_after[0]) / (p2[0]-p1[0]))
                                    if p2[1] != p1[1] :
                                        rate_w.append((p2_after[1]-p1_after[1]) / (p2[1]-p1[1]))
                                except :
                                    print('')
                            if len(rate_h) > 0 :
                                track._rate_height = np.median(np.asarray(rate_h))
                            if len(rate_w) > 0 :
                                track._rate_width = np.median(np.asarray(rate_w))

                        # the flow has been computed for re_flow frames, so the exact speed is obtained
                        # by dividing these quantities by re_flow
                        track._last_speed_from_optical_flow /= re_flow
                        track._speed = track._last_speed_from_optical_flow
                    else :
                        track._last_speed_from_optical_flow = np.zeros(shape=(4), dtype=np.float32)
                        track._mask.fill(0)
                        track._rate_height = 1.0
                        track._rate_width = 1.0
                        track._speed = np.zeros(shape=(4), dtype=np.float32)
                        for s in track._set_of_old_speed:
                            track._speed += s
                        if len(track._set_of_old_speed) > 0:
                            track._speed /= len(track._set_of_old_speed)
                            track._speed *= 1.0

    def predict(self, frame_id, re_flow) :
        if True :
            # update box by speed
            removed = []
            for index in self._tracklets :
                track = self._tracklets[index]
                if track._valid :
                    # track frame by frame
                    old_w = track._bounding_box[2] - track._bounding_box[0]
                    old_h = track._bounding_box[3] - track._bounding_box[1]
                    diff_w = old_w * (track._rate_width-1.0)
                    diff_h = old_h * (track._rate_height-1.0)
                    track._bounding_box[0] += track._speed[0]
                    track._bounding_box[1] += track._speed[1]
                    track._bounding_box[2] += track._speed[2]
                    track._bounding_box[3] += track._speed[3]
                    # adapt rectangle's size
                    track._bounding_box[0] -= diff_w / 2.0
                    track._bounding_box[1] -= diff_h / 2.0
                    track._bounding_box[2] += diff_w / 2.0
                    track._bounding_box[3] += diff_h / 2.0
                    if track._bounding_box[0] >= track._bounding_box[2]\
                        or track._bounding_box[1] >= track._bounding_box[3] :
                        track._valid = False
                        removed.append(index)

            self.remove_abandonned_tracks(removed)

    def show_on_image(self, image, frame_id, set_of_ids=None, show_mask=False, show_arrow=False) :
        for id_track in self._tracklets :
            if set_of_ids is None or id_track in set_of_ids :
                track = self._tracklets[id_track]
                if track.valid_to_show(frame_id) :
                    color_rectangle = track._color_vis.tolist()
                    if show_mask and track._mask is not None :
                        image[track._mask == 1] = track._color_vis
                    cv2.rectangle(image, (int(track._bounding_box[0]),
                                          int(track._bounding_box[1])),
                                  (int(track._bounding_box[2]),
                                   int(track._bounding_box[3])), color_rectangle,
                                  thickness=2)
                    cv2.putText(image, str(track._id),
                                (int(track._bounding_box[0]) + 2, int(track._bounding_box[1]) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0), thickness=2)
                    if show_arrow and track._mask is not None :
                        centre = [0, 0]
                        centre[0] = (track._bounding_box[1] + track._bounding_box[3]) / 2.0
                        centre[1] = (track._bounding_box[0] + track._bounding_box[2]) / 2.0
                        end_p = (int(centre[1] + track._speed[0]*5), int(centre[0] + track._speed[1]*5))
                        begin_p = (int(centre[1]), int(centre[0]))
                        cv2.arrowedLine(image, begin_p, end_p, color=(255, 0, 0), thickness=2)

    def save_to_txt(self, out_file, out_file_detailed, frame_id) :
        aggregate_iou = {}
        aggregate_px = {}
        aggregate_rel = {}
        lost = 0
        id_lost = []
        for id in self._error_summary :
            cur_errors = self._error_summary[id]
            first_frame = cur_errors['first_frame']
            cur_lost = cur_errors['lost']
            if cur_lost is not None :
                lost += 1
                id_lost.append(id)
            for elem in cur_errors :
                if '_iou' in elem :
                    frame = int(elem.split('_')[0]) - first_frame
                    error = cur_errors[elem]
                    if frame not in aggregate_iou:
                        updated_set = []
                    else:
                        updated_set = aggregate_iou[frame]
                    updated_set.append(error)
                    aggregate_iou[frame] = updated_set
                elif '_relative' in elem :
                    frame = int(elem.split('_')[0]) - first_frame
                    error = cur_errors[elem]
                    if frame not in aggregate_rel :
                        updated_set = []
                    else :
                        updated_set = aggregate_rel[frame]
                    updated_set.append(error)
                    aggregate_rel[frame] = updated_set
                elif '_px' in elem :
                    frame = int(elem.split('_')[0]) - first_frame
                    error = cur_errors[elem]
                    if frame not in aggregate_px :
                        updated_set = []
                    else :
                        updated_set = aggregate_px[frame]
                    updated_set.append(error)
                    aggregate_px[frame] = updated_set

        # averaging
        for frame in aggregate_iou :
            aggregate_iou[frame] = sum(aggregate_iou[frame]) / len(aggregate_iou[frame])
        for frame in aggregate_px :
            aggregate_px[frame] = sum(aggregate_px[frame]) / len(aggregate_px)
        for frame in aggregate_rel :
            aggregate_rel[frame] = sum(aggregate_rel[frame]) / len(aggregate_rel[frame])

        # save to file
        out_file.write(str(frame_id) + ' :\n')
        out_file.write(str(aggregate_iou) + '\n')
        out_file.write(str(aggregate_px) + '\n')
        out_file.write(str(aggregate_rel) + '\n')
        out_file.write('Lost : ' + str(lost) + ' / ' + str(len(self._error_summary)) + '\n')
        out_file.write('Lost ids : ' + str(id_lost) + '\n')

        # save detailed to file
        out_file_detailed.write(str(frame_id) + ' :\n')
        out_file_detailed.write(str(self._error_summary) + '\n')