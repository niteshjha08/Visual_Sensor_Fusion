import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as u

def lidar_camera_fusion(pts_lidar_3d, pts_cam_2d, bboxes, img):
    cmap = plt.cm.get_cmap("hsv",256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3] * 255
    for box in bboxes:  
      distances = []
      for i in range(len(pts_cam_2d)):
        depth = pts_lidar_3d[i,0]
        
        if(u.rectContains(box,pts_cam_2d[i,:],img.shape[1], img.shape[0],0.1)):
          distances.append(depth)
          color = cmap[int(510.0/depth),:]
          cv2.circle(
            img,(int(np.round(pts_cam_2d[i, 0])), int(np.round(pts_cam_2d[i, 1]))),2,
            color=tuple(color),
            thickness=-1)
      if len(distances)>2:
        inlier_distances = u.filter_outliers(distances)
        best_distance = u.get_best_distance(inlier_distances, "closest")
        cv2.putText(img, '{0:.2f} m'.format(best_distance),(int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA)
    return img, inlier_distances

def early_fusion_pipeline(image, point_cloud, lidar2cam, detector):
    img = image.copy()
    # Show LidAR on Image
    lidar_img = lidar2cam.show_lidar_on_image(point_cloud[:,:3], image)
    # Run obstacle detection in 2D
    result, detections = detector.detect(img)
    pred_bboxes = detections[:,1]
    # Fuse Point Clouds & Bounding Boxes
    img_final, _ = lidar_camera_fusion(lidar2cam.imgfov_pc_velo,lidar2cam.imgfov_pts_2d,pred_bboxes, result)
    return img_final