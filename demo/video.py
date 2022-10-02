#!/usr/bin/python
# -*- coding:utf8 -*-
from email.feedparser import FeedParser
import os
import os.path as osp
from pickle import FALSE
import sys

sys.path.insert(0, osp.abspath('../'))

from tqdm import tqdm
import logging

from datasets.process.keypoints_ord import coco2posetrack_ord_infer
from tools.inference import inference_PE
from object_detector.YOLOv3.detector_yolov3 import inference_yolov3
from utils.utils_folder import list_immediate_childfile_paths, create_folder
from utils.utils_video import video2images, image2video
from utils.utils_image import read_image, save_image
from utils.utils_json import write_json_to_file
from engine.core.vis_helper import add_poseTrack_joint_connection_to_image, add_bbox_in_image
from utils.utils_angle import hip_cul, csvplt, csv_angleplt, csv_cogplt, output, angleplt, angleplt_smo, angleplt_cog, coordplt, stack_coords, trandition
from utils.utils_peek import angle_peek 
from utils.utils_lumina import lumina, lumina_ex
from utils.utils_cog import cog_cul, cog_plt 
from utils.utils_calculation import rounding, round_dp2

import numpy as np
import cv2
import matplotlib.pyplot as plt



zero_fill = 8

logger = logging.getLogger(__name__)

joint_list = ['nose', 'left_eye', 'right_eye','left_ear', 'right_ear', \
               'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', \
               'left_wrist', 'right_wrist', 'left_hip', 'right_hip', \
               'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

def main():
    video()


def video():
    logger.info("Start")
    base_video_path = "./input"
    base_img_vis_save_dirs = './output/vis_img'
    json_save_base_dirs = './output/json'
    create_folder(json_save_base_dirs)
    video_list = list_immediate_childfile_paths(
        base_video_path, ext=['mp3', 'mp4'])
    input_image_save_dirs = []
    SAVE_JSON = True
    SAVE_VIS_VIDEO = True
    SAVE_VIS_IMAGE = True
    SAVE_BOX_IMAGE = True
    base_img_vis_box_save_dirs = './output/vis_img_box'
    frame_nlist = []
    hip_angle_list = []
    x_cog = []
    y_cog = []
    nec_res_list = []
    tru_res_list = []
    kne_res_list = []
    ank_res_list = []
    nec_angle_list = []
    tru_angle_list = []
    kne_angle_list = []
    ank_angle_list = []
    fps = 60

    # 1.Split the video into images

    for video_path in tqdm(video_list):
        video_name = osp.basename(video_path)
        temp = video_name.split(".")[0]
        image_save_path = os.path.join(base_video_path, temp)
        image_vis_save_path = os.path.join(base_img_vis_save_dirs, temp)
        image_vis_box_save_path = os.path.join(
            base_img_vis_box_save_dirs, temp)
        input_image_save_dirs.append(image_save_path)

        create_folder(image_save_path)
        create_folder(image_vis_save_path)
        create_folder(image_vis_box_save_path)

        video2images(video_path, image_save_path)  # jpg

    # 2. Person Instance detection
    logger.info("Person Instance detection in progress ...")
    video_candidates = {}
    for index, images_dir in tqdm(enumerate(input_image_save_dirs)):
        # if index >= 1:
        #     continue
        video_name = osp.basename(images_dir)
        image_list = list_immediate_childfile_paths(images_dir, ext='jpg')
        video_candidates_list = []
        for image_path in tqdm(image_list):
            candidate_bbox = inference_yolov3(image_path)
            for bbox in candidate_bbox:
                # bbox  - x, y, w, h
                video_candidates_list.append({"image_path": image_path,
                                              "bbox": bbox,
                                              "keypoints": None})
        video_candidates[video_name] = {"candidates_list": video_candidates_list,
                                        "length": len(image_list)}
    logger.info("Person Instance detection finish")
    # 3. Singe Person Pose Estimation
    logger.info("Single person pose estimation in progress ...")
    for video_name, video_info in video_candidates.items(): # 各画像ごとのループ
        video_candidates_list = video_info["candidates_list"]
        video_length = video_info["length"]
        prev_image_id = None
        frame_nlist = []
        x_cog = []
        y_cog = []
        angle_sum_list = []
        tru_res_list = []
        nec_res_list = []
        kne_res_list = []
        ank_res_list = []
        nec_angle_list = []
        tru_angle_list = []
        kne_angle_list = []
        ank_angle_list = []
        for person_info in tqdm(video_candidates_list):
            image_path = person_info["image_path"]
            xywh_box = person_info["bbox"]
            print(os.path.basename(image_path))
            image_idx = int(os.path.basename(image_path).replace(".jpg", ""))
            # from
            prev_idx, next_id = image_idx - 1, image_idx + 1
            if prev_idx < 0:
                prev_idx = 0
            if image_idx >= video_length - 1:
                next_id = video_length - 1
            prev_image_path = os.path.join(os.path.dirname(
                image_path), "{}.jpg".format(str(prev_idx).zfill(zero_fill)))
            next_image_path = os.path.join(os.path.dirname(
                image_path), "{}.jpg".format(str(next_id).zfill(zero_fill)))

            # current_image = read_image(image_path)
            # prev_image = read_image(prev_image_path)
            # next_image = read_image(next_image_path)
            bbox = xywh_box
            keypoints = inference_PE(
                image_path, prev_image_path, next_image_path, bbox)
            person_info["keypoints"] = keypoints.tolist()[0]
            x_n = person_info["keypoints"][0][0]
            y_n = person_info["keypoints"][0][1]
            x_s = person_info["keypoints"][5][0]
            y_s = person_info["keypoints"][5][1]
            x_h = person_info["keypoints"][11][0]
            y_h = person_info["keypoints"][11][1]
            x_k = person_info["keypoints"][13][0]
            y_k = person_info["keypoints"][13][1]
            x_a = person_info["keypoints"][15][0]
            y_a = person_info["keypoints"][15][1]
            

            frame_nlist.append(image_idx)
            # 頸部角度推移
            nec_est_list = hip_cul(x_n, y_n, x_s, y_s)
            nec_res_list.append(nec_est_list)
            nec_angle_list.append(float(round(nec_est_list[14], 2)))
            # 臀部角度推移
            tru_est_list = hip_cul(x_s, y_s, x_h, y_h)
            tru_res_list.append(tru_est_list)
            tru_angle_list.append(float(round(tru_est_list[14], 2)))
            # 膝部角度推移
            kne_est_list = hip_cul(x_h, y_h, x_k, y_k)
            kne_res_list.append(kne_est_list)
            kne_angle_list.append(float(round(kne_est_list[14], 2)))
            # 足部角度推移
            ank_est_list = hip_cul(x_k, y_k, x_a, y_a)
            ank_res_list.append(ank_est_list)
            ank_angle_list.append(float(round(ank_est_list[14], 2)))

            # 重心推移

            crr_cog = cog_cul(person_info["keypoints"])
            x_cog.append(float(round(crr_cog[0], 2)))
            y_cog.append(float(round(crr_cog[1], 2)))

            # 全座標出力
            stack_coords(angle_sum_list, person_info["keypoints"])

            # posetrack points
            new_coord = coco2posetrack_ord_infer(keypoints[0])
            # pose
            if SAVE_VIS_IMAGE:
                image_save_path = os.path.join(os.path.join(
                    base_img_vis_save_dirs, video_name), image_path.split("/")[-1])
                if osp.exists(image_save_path):
                    current_image = read_image(image_save_path)
                else:
                    current_image = read_image(image_path)
                pose_img = add_poseTrack_joint_connection_to_image(
                    current_image, new_coord, sure_threshold=0.3, flag_only_draw_sure=True)
                cv2.circle(pose_img, (int(round(crr_cog[0])), int(round(crr_cog[1]))), 7, (153, 0, 255), thickness=-1)
                save_image(image_save_path, pose_img)

            if SAVE_BOX_IMAGE:
                image_save_path = os.path.join(os.path.join(
                    base_img_vis_box_save_dirs, video_name), image_path.split("/")[-1])
                if osp.exists(image_save_path):
                    current_image = read_image(image_save_path)
                else:
                    current_image = read_image(image_path)
                xyxy_box = bbox[0], bbox[1], bbox[0] + \
                    bbox[2], bbox[1] + bbox[3]
                box_image = add_bbox_in_image(current_image, xyxy_box)
                save_image(image_save_path, box_image)

        if SAVE_JSON:
            joints_info = {"Info": video_candidates_list}
            temp = "result_" + video_name + ".json"
            write_json_to_file(joints_info, os.path.join(
                json_save_base_dirs, temp))
            print("------->json Info save Complete!")
            print("------->Visual Video Compose Start")
        if SAVE_VIS_VIDEO:
            image2video(os.path.join(
                base_img_vis_save_dirs, video_name), video_name)
            print("------->Complete!")


            cog_coords = np.array([x_cog, y_cog])
            csv_angleplt(video_name, nec_res_list, '01', 'neck') # 頸部角度出力
            csv_angleplt(video_name, tru_res_list, '02', 'hip') # 臀部角度出力
            csv_angleplt(video_name, kne_res_list, '03', 'knee') # 膝部角度出力
            csv_angleplt(video_name, ank_res_list, '04', 'ankle') # 足部角度出力
            csv_cogplt(video_name, cog_coords.transpose())
            trandition(video_name, angle_sum_list) # 座標のデータ出力
            # angle_peek(video_name, angle_list, fps) # 最大角度，最小角度の値とそのフレーム数
            angleplt_smo(video_name, frame_nlist, nec_angle_list, fps, '01', 'neck') # 角度推移のグラフ
            plt.pause(2)
            angleplt_smo(video_name, frame_nlist, tru_angle_list, fps, '02', 'hip') # 角度推移のグラフ
            plt.pause(2)
            angleplt_smo(video_name, frame_nlist, kne_angle_list, fps, '03', 'knee') # 角度推移のグラフ
            plt.pause(2)
            angleplt_smo(video_name, frame_nlist, ank_angle_list, fps, '04', 'ankle') # 角度推移のグラフ
            plt.pause(2)
            # angleplt_cog(video_name + '_cog', frame_nlist, x_cog, y_cog, fps) # 重心推移のグラフ]


                


if __name__ == '__main__':
    main()
