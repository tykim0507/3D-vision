import numpy as np

from feature import EstimateE_RANSAC


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0:  # Ensure proper rotation matrix with determinant 1
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    # Camera centers calculated as -R.T * t
    C1 = -R1.T @ t1
    C2 = -R1.T @ t2
    C3 = -R2.T @ t1
    C4 = -R2.T @ t2
    
    R_set = np.array([R1, R1, R2, R2])
    C_set = np.array([C1, C2, C3, C4])

    return R_set, C_set


def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    assert track1.shape == track2.shape, "track1 and track2 must have the same shape"
    n = track1.shape[0]

    X = -1 * np.ones((n, 3))
    for i in range(n):
        x1, y1 = track1[i]
        x2, y2 = track2[i]
        if x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1:
            continue
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])
        
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1]
        point_3d /= point_3d[-1]
        X[i] = point_3d[:-1]
        
    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    
    R1, t1 = P1[:, :3], P1[:, 3]
    R2, t2 = P2[:, :3], P2[:, 3]
    C1, C2 = -R1.T @ t1, -R2.T @ t2
    
    mask1 = np.sum(X, axis=-1) != -3
    mask2 = np.logical_and(np.dot(R1[2], (X - C1).T) > 0, np.dot(R2[2], (X - C2).T) > 0)
    
    valid_index = np.logical_and(mask1, mask2)
    

    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    # Only use the features that are visible in both images
    feature_mask = np.logical_and(np.sum(track1, axis=-1) != -2, np.sum(track2, axis=-1) != -2)
    x1, x2 = track1[feature_mask], track2[feature_mask]
    featureIdx = np.asarray(np.nonzero(feature_mask)[0])
    
    
    # Estimate the essential matrix
    E, inlier = EstimateE_RANSAC(x1, x2, 200, 0.01)
    # Get 4 possible camera poses
    R_set, C_set = GetCameraPoseFromE(E)
    
    P1 = np.eye(3, 4)
    valid_points = 0
    
    for i in range(4):
        P2 = np.hstack([R_set[i], -(R_set[i] @ C_set[i]).reshape((3, 1))])
        # Triangulate points
        X_3d = Triangulation(P1, P2, track1, track2)
        # Filter out points based on cheirality
        valid_index = EvaluateCheirality(P1, P2, X_3d)
        print(f"Valid points: {np.sum(valid_index)} for camera pose {i}")
        if np.sum(valid_index) > valid_points:
            valid_points = np.sum(valid_index)
            R = R_set[i]
            C = C_set[i]
            X = -1 * np.ones((track1.shape[0], 3))
            X[valid_index] = X_3d[valid_index]
    
    return R, C, X