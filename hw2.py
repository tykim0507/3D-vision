import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment


def check_proj_error(idx, X, track, K, P, inlier=None):
    is_X_valid = np.all(X != -1, axis=1)
    if inlier is not None:
        is_inlier = np.zeros_like(is_X_valid)
        inlier_idx = np.where(inlier)[0]
        is_inlier[inlier_idx] = True
        is_X_valid = np.logical_and(is_X_valid, is_inlier)
    X_valid = X[is_X_valid]
    X_valid_h = np.hstack([X_valid, np.ones((X_valid.shape[0], 1))])
    x_proj_h = P[idx] @ X_valid_h.T
    x_proj = x_proj_h[:2] / x_proj_h[2]
    x_valid = track[idx][is_X_valid].T
    
    x_proj_K = K @ x_proj_h
    x_proj_K = x_proj_K[:2] / x_proj_K[2]
    x_proj_K = x_proj_K.T
    x_valid_K = K @ np.vstack([x_valid, np.ones((1, x_valid.shape[1]))])
    x_valid_K = x_valid_K[:2] / x_valid_K[2]    
    x_valid_K = x_valid_K.T
    print("Ideal                        Projected")
    print(np.hstack([x_valid_K[:20], x_proj_K[:20]]))
    print(np.linalg.norm(x_proj_K - x_valid_K))

    return


if __name__ == '__main__':
    np.random.seed(100)
    K = np.asarray([
        [463.1, 0, 333.2],
        [0, 463.1, 187.5],
        [0, 0, 1]
    ])
    num_images = 14
    w_im = 672
    h_im = 378

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = 'images/image{:d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)
    print("Track building over")
    
    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)
    
    output_dir = 'output1'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))
    # Set first two camera poses
    P[0] = np.eye(3, 4)
    P[1] = np.hstack([R, -(R @ C).reshape((3, 1))])
    # print("camera 1, 2 initialized")
    # print("checking projection error")
    # check_proj_error(1, X, track, K, P)
    ransac_n_iter = 500
    ransac_thr = 0.01
    for i in range(2, num_images):
        print(f"adding camera number {i+1}")
        # Estimate new camera pose
        track_i = track[i, :, :]
        
        
        R, C, inlier = PnP_RANSAC(X, track_i, ransac_n_iter, ransac_thr)
        inlier_idx = np.where(inlier)[0]
        # breakpoint()
        print(f"number of inliers for camera{i+1}: {len(inlier_idx)}")
        R, C = PnP_nl(R, C, X[inlier_idx, :], track_i[inlier_idx, :])

        P[i] = np.hstack([R, -(R @ C).reshape((3, 1))])
           
        for j in range(i):
            print(f"matching camera {j+1} with {i+1}")
            # Fine new points to reconstruct
            track_j = track[j, :, :]
            track_j_mask = np.logical_and(track_j[:, 0] != -1, track_j[:, 1] != -1)
            
            new_points = np.logical_and(FindMissingReconstruction(X, track_i), track_j_mask)
            new_points_idx = np.where(new_points)[0]
            print(f"new points added with camera {i+1} in iteration {j+1}: {len(new_points_idx)}")
            # Triangulate points
            new_X = Triangulation(P[i], P[j], track_i[new_points, :], track_j[new_points, :])
            new_X = Triangulation_nl(new_X, P[i], P[j], track_i[new_points, :], track_j[new_points, :])
            
            # Filter out points based on cheirality
            valid_idx = EvaluateCheirality(P[i], P[j], new_X)
            # Update 3D points
            X[new_points_idx[valid_idx], :] = new_X[valid_idx, :]
        # Run bundle adjustment
        
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        P[:i + 1, :, :] = P_new
        X[valid_ind, :] = X_new

        P[:i+1,:,:] = P_new
        X[valid_ind,:] = X_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)
        
        print(f"checking projection error of {i+1}th camera")
        check_proj_error(i, X, track, K, P, inlier=inlier)