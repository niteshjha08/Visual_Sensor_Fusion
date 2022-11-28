import numpy as np
import cv2
import matplotlib.pyplot as plt

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, [3, 3])

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # ignore non-floats
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def get_pcd_projection_on_image(self, pts_3d_velo):
        '''
        Input: 3D points in LiDAR Frame [nx3]
        Output: 2D Pixels in Image Frame [nx2]
        '''
        
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        p_r0 = np.dot(self.P, R0_homo_2) #PxR0
        p_r0_rt =  np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1]))) #PxROxRT
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0],1))])
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))#PxROxRTxX
        pts_2d = np.transpose(p_r0_rt_x)
        
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def filter_pcd_in_image_fov(self,pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pts_2d = self.get_pcd_projection_on_image(pc_velo)
        fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) # We don't want things that are closer to the clip distance (2m)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo

    def show_lidar_on_image(self, pc_velo, img):
        """ Project LiDAR points to image """
        imgfov_pc_velo, pts_2d, fov_inds = self.filter_pcd_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], True
        )

        self.imgfov_pts_2d = pts_2d[fov_inds, :]
        
        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        self.imgfov_pc_velo = imgfov_pc_velo
        
        for i in range(self.imgfov_pts_2d.shape[0]):
            depth = imgfov_pc_velo[i,0]
            color = cmap[int(510.0 / depth), :]
            cv2.circle(
                img,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,
                color=tuple(color),
                thickness=-1,
            )

        return img

