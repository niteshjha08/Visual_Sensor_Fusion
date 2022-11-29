import numpy as np
from Lidar2Camera import LiDAR2Camera
from yolov4 import YOLOv4
from fusion import early_fusion_pipeline
import glob
import cv2
import open3d as o3d
import config as cfg

def visualize_image_and_pcd():
    image_files = sorted(glob.glob("./../data/img/*.png"))
    point_files = sorted(glob.glob("./../data/velodyne/*.pcd"))
    label_files = sorted(glob.glob("./../data/label/*.txt"))
    calib_files = sorted(glob.glob("./../data/calib/*.txt"))

    index = 0
    lidar2cam = LiDAR2Camera(calib_files[index])
    image = cv2.imread(image_files[index])

    pcd_file = point_files[index]
    # cv2.imshow('img',image)
    # cv2.waitKey(0)

    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.array(pcd.points)
    o3d.visualization.draw_geometries([pcd])
    cv2.imshow('image_before',image)

    result = lidar2cam.show_lidar_on_image(points, image.copy())
    cv2.imshow('result',result)
    cv2.imshow('image',image)
    cv2.waitKey(0)

def test_early_fusion():
    image_names = sorted(glob.glob(cfg.images_path + "*.png"))

    point_files = sorted(glob.glob(cfg.points_path + "*.pcd"))
    calib_files = sorted(glob.glob(cfg.calib_path))

    index = 0
    lidar2cam = LiDAR2Camera(calib_files[index])
    
    image = cv2.imread(image_names[0])
    pcd_file = point_files[index]

    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.array(pcd.points)

    detector = YOLOv4()
    detector.load_model(cfg.weights_file, cfg.config_file, cfg.names_file)

    result = early_fusion_pipeline(image, points, lidar2cam, detector)
    cv2.imshow('res',result)
    cv2.waitKey(0)


def test_detection():
    image_files = sorted(glob.glob(cfg.images_path + "*.png"))
    names_file = "./../data/yolo/coco.names"
    weights_file = "./../data/yolo/yolov4.weights"
    config_file = "./../data/yolo/yolov4.cfg"

    print(image_files[0])
    index = 0
    image = cv2.imread(image_files[0])
    cv2.imshow('img', image)
    detector = YOLOv4()
    detector.load_model(weights_file,config_file,names_file)
    result,_ = detector.detect(image.copy())
    cv2.imshow('res',result)
    cv2.waitKey(0)