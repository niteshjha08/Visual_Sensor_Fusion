# Visual_Sensor_Fusion
Low-level fusion and Mid-level fusion of point clouds and images

## Steps to run
1. Place images sequence (dir nam images) and pcd file (dir name points) in <input_dir>.
2. Set data_path in main.py as <input_dir>
3. Download YOLOv4 weights from [here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and place it inside yolo dir
4. Run main.py

## Low level fusion
1. YOLO detections on video

![Detections](https://github.com/niteshjha08/Visual_Sensor_Fusion/blob/main/data/media/detections_1.gif)

2. LiDAR points projection on image

![LiDAR projection](https://github.com/niteshjha08/Visual_Sensor_Fusion/blob/main/data/media/lidar_1.gif)

3. Fusion of LiDAR projections and YOLO detections

![Fusion](https://github.com/niteshjha08/Visual_Sensor_Fusion/blob/main/data/media/ouput1.gif)
