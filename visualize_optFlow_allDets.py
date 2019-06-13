import os
import numpy as np
import cv2
from show_flo import readFlow
from tracking_from_det import Tracker_from_det
import argparse, time, copy
from datasets.mot_seq import get_loader

profiling = False
# profiling
if profiling :
    import cProfile

#=============================== PARAMETERS TO BE MODIFIED =====================================#

#frames_dir = '//10.126.40.37/jafar/Khanh/netdef_models/FlowNet3/image_test/levi_extr.mp4'
#frames_dir = '//10.126.40.35/jafar/Khanh/image_test/levi_extr_r2.mp4'
frames_dir = './data/frames/20190325_172557.mp4'

first_frame = 0
#re_det = 5
re_det = 1
re_flow = 5
show_or_save_baseline = False
show_or_save_flow = True
show = True
save = 'image' # 'image' or 'video'
save_ratio = 1.0
save_to_txt = True
threshold_iou = 1.0
threshold_on_standby_frames = 25
#threshold_on_standby_frames_to_view = threshold_on_standby_frames
threshold_on_standby_frames_to_view = re_det + 3
threshold_on_standby_frames_to_save = re_det + 3
#initializing_frames = 3
initializing_frames = 2
buffer_size = 3
show_mask = False
show_arrow = True

# specific to optical flow
#flo_dir = '//10.126.40.37/jafar/Khanh/netdef_models/FlowNet3/css_small/output/levi_extr_r2.mp4'
#flo_dir = '//10.126.40.39/jafar/Khanh/out_optflow_5/street_overview.mp4_f5_flownet_css_default_noresize'
#flo_dir = '//10.126.40.45/jafar/SSD/Khanh/output_optflow/960_540/flow5/shutterstock_v10199435.mov_f5_flownet_css_default_noresize'
flo_dir = None
#flo_dir_bw = '//10.126.40.35/jafar/Khanh/out/levi_extr.mp4_flownet_cs_train_FlyingChairs_noanydataaug_short_ftfromCS.py_backward'
#flo_dir = None
flo_dir_bw = None

combined = 'only_on_objects' # 'only_on_objects' or 'all'
# if combined == 'all', uses backward flow to adjust forward flow on all points of the image
# if combined == 'only_on_objects', uses backward flow to adjust forward one only on points inside
# detected objects.

# an experimental option
save_position = False

# detections
#orig_det_fn = '../det/20190408_094024_ped.avi.txt'
orig_det_fn = './data/dets/20190325_172557_ped.mp4.txt'
# the frames extracted have been downsized, so we need to adapt the det result
force_w = 960
force_h = 540
ratio_compress_width = 0.0 # not to change, that will be computed from force_w and force_h
ratio_compress_height = 0.0 # not to change, that will be computed from force_w and force_h
output_dir_prefix = './data/output'

do_estimate_mask=False
adapt_size=False

#===============================================================================================#

computing_time = {
    # computing time for each function used in tracking
    # predict : called at each frame
    # matching : called once every re_det frames
    # estimate_track_speed : called once every re_flow frames
    'predict' : None,
    'matching' : None,
    'estimate_track_speed' : None,
}
std_item = {'total': 0, 'count_frame': 0, 'average_frame': 0, 'count_obj': 0, 'average_obj': 0}
for it in computing_time :
    computing_time[it] = copy.deepcopy(std_item)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=False, default=frames_dir)
    parser.add_argument('--first_frame', type=int, required=False, default=first_frame)
    parser.add_argument('--re_det', type=int, required=False, default=re_det)
    parser.add_argument('--re_flow', type=int, required=False, default=re_flow)
    parser.add_argument('--show', type=int, required=False, default=int(show))
    parser.add_argument('--save', type=str, required=False, default=save)
    parser.add_argument('--save_ratio', type=float, required=False, default=save_ratio)
    parser.add_argument('--threshold_iou', type=float, required=False, default=threshold_iou)
    parser.add_argument('--threshold_on_standby_frames', type=int, required=False, default=threshold_on_standby_frames)
    parser.add_argument('--threshold_on_standby_frames_to_view', type=int, required=False, default=threshold_on_standby_frames_to_view)
    parser.add_argument('--threshold_on_standby_frames_to_save', type=int, required=False, default=threshold_on_standby_frames_to_save)
    parser.add_argument('--initializing_frames', type=int, required=False, default=initializing_frames)
    parser.add_argument('--buffer_size', type=int, required=False, default=buffer_size)
    parser.add_argument('--flo_dir', type=str, required=False, default=flo_dir)
    parser.add_argument('--orig_det_fn', type=str, required=False, default=orig_det_fn)
    FLAGS = parser.parse_args()

    # update parameters if necessary
    frames_dir = FLAGS.frames_dir
    first_frame = FLAGS.first_frame
    re_det = FLAGS.re_det
    re_flow = FLAGS.re_flow
    show = FLAGS.show
    save = FLAGS.save
    save_ratio = FLAGS.save_ratio
    threshold_iou = FLAGS.threshold_iou
    threshold_on_standby_frames = FLAGS.threshold_on_standby_frames
    threshold_on_standby_frames_to_view = FLAGS.threshold_on_standby_frames_to_view
    threshold_on_standby_frames_to_save = FLAGS.threshold_on_standby_frames_to_save
    initializing_frames = FLAGS.initializing_frames
    buffer_size = FLAGS.buffer_size
    flo_dir = FLAGS.flo_dir
    orig_det_fn = FLAGS.orig_det_fn

    if flo_dir is not None :
        output_dir = output_dir_prefix + '/' + os.path.basename(flo_dir)
    else :
        output_dir = output_dir_prefix + '/' + os.path.basename(frames_dir)
    output_dir += '_reflow_' + str(re_flow) + '_redet_' + str(re_det)
    if flo_dir_bw is not None :
        output_dir += '_bidir_' + combined
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    output_txt_fn = output_dir + '/det.txt'
    if save_to_txt :
        output_txt_f = open(output_txt_fn, 'w')

    # look for resolution of input images
    img = cv2.imread(frames_dir + '/frame0.png')
    if img is not None:
        w_img = img.shape[1]
        h_img = img.shape[0]
    else:
        print(frames_dir + '/frame0.png does not exist, the resolution of the output video can not be identified!')
        exit(1)

    if force_w == 0 or force_h == 0 :
        # take the original dimensions
        w = w_img
        h = h_img
    else :
        w = force_w
        h = force_h
        ratio_compress_width = force_w / w_img
        ratio_compress_height = force_h / h_img

    if save == 'video' :
        output_video_fn = output_dir + '/video.avi'
        output_video_cap = cv2.VideoWriter(output_video_fn, -1, 15.0, (int(w*save_ratio), int(h*save_ratio)))

    if os.path.exists(orig_det_fn) :
        det_data = np.fromfile(file=orig_det_fn, dtype=float, sep=' ')
        det_data = np.reshape(det_data, [-1, 10])
        filt_det_data = det_data[det_data[:, 0] >= first_frame]
        # resize to the new resolution
        if ratio_compress_width > 0.0 :
            filt_det_data[:, 2] *= ratio_compress_width
            filt_det_data[:, 4] *= ratio_compress_width
        if ratio_compress_height > 0.0 :
            filt_det_data[:, 3] *= ratio_compress_height
            filt_det_data[:, 5] *= ratio_compress_height
        det_data = None

        # regroup data by frame
        dict_data = {}
        for r_data in filt_det_data :
            index = int(r_data[0])
            if index not in dict_data :
                cur_dict_data = []
            else :
                cur_dict_data = dict_data[index]
            cur_dict_data.append(r_data)
            dict_data[index] = cur_dict_data

        min_frame = int(filt_det_data[0][0])
        max_frame = int(filt_det_data[-1][0])
        #max_frame = 100
        filt_det_data = None

        # show onto images
        count = 0
        logBy = 1
        # initialize structure for each identity
        tracker = Tracker_from_det(threshold_iou = threshold_iou,min_ap_dist=0.64,max_n_features =1000, initializing_frames=initializing_frames,
                                   threshold_on_standby_frames=threshold_on_standby_frames,
                                   threshold_on_standby_frames_to_view=threshold_on_standby_frames_to_view,
                                   threshold_on_standby_frames_to_save=threshold_on_standby_frames_to_save,
                                   buffer_size=buffer_size)

        # if profiling
        if profiling :
            pr = cProfile.Profile()
        for frame_id in range(min_frame, max_frame) :
            if frame_id == 57:
                print('stop')
            if frame_id in dict_data :
                array_r_data = dict_data[frame_id]
            else :
                array_r_data = None

            image_fn = frames_dir + '/frame' + str(frame_id) + '.png'
            image = cv2.imread(image_fn)
            if ratio_compress_width > 0 and ratio_compress_height > 0 :
                image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
            if image is not None :
                if show_or_save_baseline :
                    if array_r_data is not None :
                        for r_data in array_r_data :
                            box = [r_data[2], r_data[3], r_data[4], r_data[5]]
                            cv2.rectangle(image, (int(box[0]), int(box[1])),
                                        (int(box[0] + box[2]), int(box[1] + box[3])), (255, 0, 0),
                                        thickness=2)

                if frame_id == min_frame :
                    # first frame, just update the initial positions
                    if array_r_data is not None :
                        # first frame, just update box position
                        tracker.add_new_tracks(dets_data=array_r_data,
                                               new_dets=range(len(array_r_data)),
                                               frame_id=frame_id, image = image)
                    else :
                        print('Bug array_r_data; exit')
                        exit(1)
                elif re_det > 0 and count%re_det == 0 :
                    # TODO : reimplement tracking with openCV
                    time_start = time.clock()
                    tracker.predict(frame_id, re_flow)
                    time_elapsed = time.clock() - time_start
                    time_elapsed *= 1000
                    computing_time['predict']['total'] += time_elapsed
                    computing_time['predict']['count_frame'] += 1
                    if array_r_data is not None :
                        computing_time['predict']['count_obj'] += len(array_r_data)
                    if computing_time['predict']['count_frame'] > 0 :
                        computing_time['predict']['average_frame'] = computing_time['predict']['total'] / \
                                                                     computing_time['predict']['count_frame']
                    if computing_time['predict']['count_obj'] > 0 :
                        computing_time['predict']['average_obj'] = computing_time['predict']['total'] / \
                                                                   computing_time['predict']['count_obj']
                    if array_r_data is not None :
                        # use detected boxes to update the tracking
                        time_start = time.clock()
                        matches = tracker.matching_score_iou(array_r_data)
                        #matche = tracker.matching_reid(matches)
                        matches, abandonned_tracks, new_dets = \
                            tracker.refine_matching_score(matches, array_r_data)
                        tracker.update_tracks(array_r_data, matches, frame_id, image)
                        index_new = tracker.add_new_tracks(array_r_data, new_dets, frame_id, image)
                        #TODO : extrack feature cua hai cai update +NEW TRACK
                        #tracker.extract_reid_feature(image)


                        #if len(abandonned_tracks)> 0:
                        set_of_abandonned_index = []
                        #set_of_negative_index = [index for index in tracker._tracklets if
                         #                        tracker._tracklets[index]._id < 0]
                        set_of_abandonned_index = abandonned_tracks
                        #set_of_abandonned_index = [m[0] for m in matches]
                        #tracker.extract_reid_feature(set_of_abandonned_index, image)
                        match = tracker.matching_reid(index_new, set_of_abandonned_index)
                        match, abandonned_track, new_det = tracker.refine_matching_reid(match)
                        #for m in match:
                            #index_new.remove(m[1])
                        tracker.merge_tracks_reid(match, frame_id, image)

                        #elif len(abandonned_tracks)==0:
                        tracker.ready_for_new_id(None)
                        time_elapsed = time.clock() - time_start
                        time_elapsed *= 1000
                        computing_time['matching']['total'] += time_elapsed
                        computing_time['matching']['count_frame'] += 1
                        if array_r_data is not None :
                            computing_time['matching']['count_obj'] += len(array_r_data)
                        if computing_time['matching']['count_frame'] > 0 :
                            computing_time['matching']['average_frame'] = computing_time['matching']['total'] / \
                                                                          computing_time['matching']['count_frame']
                        if computing_time['matching']['count_obj'] > 0 :
                            computing_time['matching']['average_obj'] = computing_time['matching']['total'] / \
                                                                        computing_time['matching']['count_obj']
                else :
                    # use solely optical flow to track objects
                    # TODO : reimplement tracking with openCV
                    time_start = time.clock()
                    tracker.predict(frame_id, re_flow)
                    time_elapsed = time.clock() - time_start
                    time_elapsed *= 1000
                    computing_time['predict']['total'] += time_elapsed
                    computing_time['predict']['count_frame'] += 1
                    if array_r_data is not None:
                        computing_time['predict']['count_obj'] += len(array_r_data)
                    if computing_time['predict']['count_frame'] > 0 :
                        computing_time['predict']['average_frame'] = computing_time['predict']['total'] / \
                                                                     computing_time['predict']['count_frame']
                    if computing_time['predict']['count_obj'] > 0 :
                        computing_time['predict']['average_obj'] = computing_time['predict']['total'] / \
                                                                   computing_time['predict']['count_obj']

                # in all case, update old_frame_id and current_speed
                tracker.review_all_tracks(frame_id)
                if frame_id%re_flow == 0 :
                    # specific to optical flow
                    if profiling :
                        pr.enable()
                    time_start = time.clock()
                    tracker.estimate_track_speed_from_flow(frame_id=frame_id,
                                                           re_flow=re_flow,
                                                           flo_dir=flo_dir,
                                                           do_estimate_mask=do_estimate_mask,
                                                           adapt_size=adapt_size)
                    time_elapsed = time.clock() - time_start
                    time_elapsed *= 1000
                    computing_time['estimate_track_speed']['total'] += time_elapsed
                    computing_time['estimate_track_speed']['count_frame'] += 1
                    if array_r_data is not None :
                        computing_time['estimate_track_speed']['count_obj'] += len(array_r_data)
                    if computing_time['estimate_track_speed']['count_frame'] > 0 :
                        computing_time['estimate_track_speed']['average_frame'] = computing_time['estimate_track_speed']['total'] / \
                                                                                  computing_time['estimate_track_speed']['count_frame']
                    if computing_time['estimate_track_speed']['count_obj'] > 0 :
                        computing_time['estimate_track_speed']['average_obj'] = computing_time['estimate_track_speed']['total'] / \
                                                                                computing_time['estimate_track_speed']['count_obj']
                    if profiling :
                        pr.disable()
                        pr.print_stats(sort='calls')
                # in the case flow is estimated, save the current position in order to track in frame (T+re_flow)
                if save_position :
                    tracker.save_position(re_flow)

                if show_or_save_flow :
                    tracker.show_on_image(image, frame_id, None, show_mask, show_arrow)
                    image = cv2.resize(image, None, None, save_ratio, save_ratio)

                if show :
                    cv2.imshow('Video', image)
                    cv2.waitKey(10)
                if save == 'image' :
                    # at the end, write to file
                    cv2.imwrite(output_dir + '/frame' + str(frame_id) + '.png', image)
                if save == 'video' :
                    # write to video
                    output_video_cap.write(image)
                if save_to_txt :
                    tracker.save_to_txt(output_txt_f, frame_id)

                image = None

            count += 1
            if count%logBy == 0 :
                print(str(count) + ' ... processed')
            print('estimate_track_speed : ' + str(computing_time['estimate_track_speed']['average_frame']))
            print('By object : ' + str(computing_time['estimate_track_speed']['average_obj']))
            print('predict : ' + str(computing_time['predict']['average_frame']))
            print('By object : ' + str(computing_time['predict']['average_obj']))
            print('matching : ' + str(computing_time['matching']['average_frame']))
            print('By object : ' + str(computing_time['matching']['average_obj']))

        if save_to_txt :
            output_txt_f.close()

        if save == 'video' :
            output_video_cap.release()

        # write info to txt file
        tracker.print_info(output_dir + '/info.txt')
        out_time_fn = output_dir + '/time.txt'
        out_time_f = open(out_time_fn, 'aw')
        out_time_f.write(str(computing_time))
        out_time_f.close()
        if profiling :
            pr.print_stats(sort='calls')