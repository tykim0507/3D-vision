import numpy as np

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    n = X.shape[0]
    
    # Building the matrix A for AX = 0 formulation
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X_i = X[i]
        x_i = x[i]
        A[2*i] = [X_i[0], X_i[1], X_i[2], 1, 0, 0, 0, 0, -x_i[0]*X_i[0], -x_i[0]*X_i[1], -x_i[0]*X_i[2], -x_i[0]]
        A[2*i+1] = [0, 0, 0, 0, X_i[0], X_i[1], X_i[2], 1, -x_i[1]*X_i[0], -x_i[1]*X_i[1], -x_i[1]*X_i[2], -x_i[1]]

    # Using SVD to solve AX = 0
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  # Reshape the last row of V to form P

    # Decompose P to get R and C
    M = P[:, :3]  # Extract the first 3 columns
    C = P[:, 3]  # Extract the last column
    
    # Ensure M is a rotation matrix by making it orthogonal
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    
    # Camera center C can be found by -R.T @ t
    t = C
    t /= S[0]
    C = -R.T @ t
    
    proj = (X - C[None]) @ R.T
    sign = np.sign(proj[:, 2])
    if np.sum(sign) < 0:
        R, C = -R, -C
    

    return R, C



def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    max_inliers = np.zeros(X.shape[0])
    best_R = None
    best_C = None
    
    X_mask = np.logical_and(np.logical_and(X[:, 0] != -1, X[:, 1] != -1), X[:, 2] != -1)
    track_i_mask = np.logical_and(x[:, 0] != -1, x[:, 1] != -1)
    mask = np.logical_and(X_mask, track_i_mask)
    mask_idx = np.where(mask)[0]
    masked_X = X[mask]
    masked_x = x[mask]

    for _ in range(ransac_n_iter):
        # Randomly sample subset of points for hypothetical inliers
        indices = np.random.choice(len(masked_X), size=6, replace=False)  # Need 6 points to perform linear matching
        R_estimate, C_estimate = PnP(masked_X[indices], masked_x[indices])
        
        # Project 3D points to 2D using the estimated pose and calculate error
        proj = (masked_X - C_estimate) @ R_estimate.T
        errors = np.sqrt((proj[:, 0] / proj[:, 2] - masked_x[:, 0])**2 + (proj[:, 1] / proj[:, 2] - masked_x[:, 1])**2)

        inliers = np.zeros(X.shape[0])
        inliers[mask_idx] = errors < ransac_thr
        
        
        if np.sum(inliers) > np.sum(max_inliers):
            max_inliers = inliers
            best_C = C_estimate
            best_R = R_estimate


    R, C = best_R, best_C
    inlier = max_inliers
    return R, C, inlier



def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = -R[0,:]
    dv_dc = -R[1,:]
    dw_dc = -R[2,:]
    # df_dc is in shape (2, 3)
    df_dc = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    # du_dR = np.concatenate([X-C, np.zeros(3), X-C])
    # dv_dR = np.concatenate([np.zeros(3), X-C, X-C])
    # dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    du_dR = np.concatenate([X-C, np.zeros(3), np.zeros(3)])
    dv_dR = np.concatenate([np.zeros(3), X-C, np.zeros(3)])
    dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    # df_dR is in shape (2, 9)
    df_dR = np.stack([
        (w * du_dR - u * dw_dR) / (w**2),
        (w * dv_dR - v * dw_dR) / (w**2)
    ], axis=0)


    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # dR_dq is in shape (9, 4)
    dR_dq = np.asarray([
        [0, 0, -4*qy, -4*qz],
        [-2*qz, 2*qy, 2*qx, -2*qw],
        [2*qy, 2*qz, 2*qw, 2*qx],
        [2*qz, 2*qy, 2*qx, 2*qw],
        [0, -4*qx, 0, -4*qz],
        [-2*qx, -2*qw, 2*qz, 2*qy],
        [-2*qy, 2*qz, -2*qw, 2*qx],
        [2*qx, 2*qw, 2*qz, 2*qy],
        [0, -4*qx, -4*qy, 0],
    ])

    dfdp = np.hstack([df_dc, df_dR @ dR_dq])

    return dfdp


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    n = X.shape[0]
    q = Rotation2Quaternion(R)

    p = np.concatenate([C, q])
    n_iters = 20
    lamb = 1
    error = np.empty((n_iters,))
    for i in range(n_iters):
        R_i = Quaternion2Rotation(p[3:])
        C_i = p[:3]

        proj = (X - C_i[np.newaxis,:]) @ R_i.T
        proj = proj[:,:2] / proj[:,2,np.newaxis]

        H = np.zeros((7,7))
        J = np.zeros(7)
        for j in range(n):
            dfdp = ComputePoseJacobian(p, X[j,:])
            H = H + dfdp.T @ dfdp
            J = J + dfdp.T @ (x[j,:] - proj[j,:])
        
        delta_p = np.linalg.inv(H + lamb*np.eye(7)) @ J
        p += delta_p
        p[3:] /= np.linalg.norm(p[3:])

        error[i] = np.linalg.norm(proj - x)


    R_refined = Quaternion2Rotation(p[3:])
    C_refined = p[:3]
    return R_refined, C_refined