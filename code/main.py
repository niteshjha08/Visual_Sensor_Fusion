#!/usr/bin/python3
import numpy as np
from Lidar2Camera import LiDAR2Camera
from yolov4 import YOLOv4
from fusion import early_fusion_pipeline
import glob
import cv2
import open3d as o3d
import config as cfg


def process_video_early_fusion(data_path, output_path, write_output = False):

    images_path = sorted(glob.glob(data_path + "images/*.png"))
    point_files = sorted(glob.glob(data_path + "points/*.pcd"))
    calib_files = sorted(glob.glob(cfg.calib_path))

    lidar2cam = LiDAR2Camera(calib_files[0])

    image_dims = cv2.imread(images_path[0]).shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 20, (image_dims[1], image_dims[0]))
    
    for i in range(len(images_path)):
        image = cv2.imread(images_path[i])
        pcd_file = point_files[i]

        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.array(pcd.points)

        detector = YOLOv4()
        detector.load_model(cfg.weights_file, cfg.config_file, cfg.names_file)

        result = early_fusion_pipeline(image, points, lidar2cam, detector)
        cv2.imshow('result',result)
        cv2.waitKey(1)

        if write_output: 
            writer.write(result)

    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    data_path = "./../data/video2/"
    output_path = "./../data/output/output7.avi"
    write_output = True
    process_video_early_fusion(data_path, output_path, write_output)