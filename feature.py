import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    # Initialize NearestNeighbors instances for both sets of descriptors
    nn1 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des1)
    nn2 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)
    
    # set the ratio threshold for the ratio test
    ratio_threshold = 0.9
    
    
    # Find the 2 nearest neighbors in image 2 for each keypoint in image 1
    distances1, indices1 = nn2.kneighbors(des1) # shape of (n1, 2)

    # Apply the ratio test for image 1 keypoints
    ratio_mask1 = distances1[:, 0] < ratio_threshold * distances1[:, 1]

    # Find the 2 nearest neighbors in image 1 for each keypoint in image 2
    distances2, indices2 = nn1.kneighbors(des2) # shape of (n2, 2)
    
    # Apply the ratio test for image 2 keypoints
    ratio_mask2 = distances2[:, 0] < ratio_threshold * distances2[:, 1]
    
    # Filter matches based on the ratio test
    matches1 = np.where(ratio_mask1)[0] # indices of the keypoints in image 1 that pass the ratio test
    matches2 = np.where(ratio_mask2)[0] # indices of the keypoints in image 2 that pass the ratio test

    
    # Ensure bidirectional consistency
    # This means for a match to be valid, the nearest neighbor of a keypoint in one image
    # must be the same keypoint in the other image that considers the first keypoint as its nearest neighbor.
    good_matches1 = []
    good_matches2 = []
    good_indices1 = []
    
    for m1 in matches1:
        m2 = indices1[m1, 0]  # Best match in image 2 for keypoint m1 in image 1
        if ratio_mask2[m2] and indices2[m2, 0] == m1:  # Check if the best match in image 1 for keypoint m2 is indeed m1, and m2 passes the ratio test
            good_matches1.append(m1)
            good_matches2.append(m2)
            good_indices1.append(m1) #append to the list of matched indices of x1 in loc1
    
    # Gather the matched keypoints locations based on the filtered indices
    
    x1 = loc1[good_matches1]
    x2 = loc2[good_matches2]
    ind1 = np.array(good_indices1)

    return x1, x2, ind1



def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    
    # We perform 8 point algorithm
    
    #construct the matrix A
    A = []
    for p1, p2 in zip(x1, x2):
        x, y = p1
        xp, yp = p2
        A.append([xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)
    # Enforce the rank 2 constraint
    U, S, Vt = np.linalg.svd(E)
    S = np.array([1, 1, 0])
    E = U @ np.diag(S) @ Vt
    return E




def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    
    max_inliers = []
    best_E = None
    
    for _ in range(ransac_n_iter):
        # randomly select 8 pairs of matched keypoints to perform 8 points algorithm
        indices = np.random.choice(x1.shape[0], 8, replace=False)
        E = EstimateE(x1[indices], x2[indices])
        # calculate the error for each pair of matched keypoints
        x1_homogeneous = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_homogeneous = np.hstack((x2, np.ones((x2.shape[0], 1))))
     
        errors = np.abs(np.diag(x2_homogeneous @ E @ x1_homogeneous.T))
        
        inliers = np.where(errors < ransac_thr)[0]
        
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_E = E
        
    E = best_E
    inlier = max_inliers
        
    return E, inlier

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c, _ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img3 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img3 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img4 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img3,img4

def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    
    print("Building feature track")
    
    # K[2][2] = 1
    assert K[2][2] == 1, "The last element of the last row of K must be 1.0."
    
    loc_list = []
    des_list = []
    
    N = Im.shape[0]
    sift = cv2.xfeatures2d.SIFT_create()
    #Extract SIFT features for each image
    print("Extracting SIFT features")
    for i in range(N):
        kp, des = sift.detectAndCompute(Im[i], None)
        loc = np.array([loc.pt for loc in kp])
        loc_list.append(loc)
        des_list.append(des)
        print(f"Extracted {loc.shape[0]} SIFT features from image {i+1}")
    
    track = np.empty((N, 0, 2))
    
    for i in range(N):
        track_i = -1 * np.ones((N, loc_list[i].shape[0], 2))
        for j in range(i+1, N):
            x1, x2, ind1 = MatchSIFT(loc_list[i], des_list[i], loc_list[j], des_list[j])
            print(f'Found {x1.shape[0]} matches between image {i+1} and image {j+1}')
            #map to normalized coordinate by multiplying K inverse
            x1 = np.dot(np.linalg.inv(K), np.vstack((x1.T, np.ones(x1.shape[0])))).T
            x2 = np.dot(np.linalg.inv(K), np.vstack((x2.T, np.ones(x2.shape[0])))).T
            #drop the last column to get the normalized coordinates
            x1 = x1[:, :2]
            x2 = x2[:, :2]
            E, inlier_indices = EstimateE_RANSAC(x1, x2, 500, 0.01)
            feature_indices = ind1[inlier_indices] #get the feature indices which are considered as inliers
            
            #update the track_i so that the inliers are stored in the correct indices
            track_i[i, feature_indices, :] = x1[inlier_indices]
            track_i[j, feature_indices, :] = x2[inlier_indices]

            import matplotlib.pyplot as plt
            F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            pt1 = np.int32(K @ np.vstack((x1[inlier_indices].T, np.ones(x1[inlier_indices].shape[0]))))
            pt2 = np.int32(K @ np.vstack((x2[inlier_indices].T, np.ones(x2[inlier_indices].shape[0]))))
            pt1 = pt1[:2].T
            pt2 = pt2[:2].T
            lines1 = cv2.computeCorrespondEpilines(pt2.reshape(-1,1,2), 2,F)
            # breakpoint()
            I = Im.copy()
            lines1 = lines1.reshape(-1,3)
            img5,img6 = drawlines(I[i],I[j],lines1,pt1,pt2)
            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1,1,2), 1,F)
            lines2 = lines2.reshape(-1,3)
            img3,img4 = drawlines(I[j],I[i],lines2,pt2,pt1)
            plt.subplot(121),plt.imshow(img5)
            plt.title(f'epipolar line of image{i+1} for image {j+1}', fontsize=10), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img3)
            plt.title(f'epipolar line of image{j+1} for image {i+1}', fontsize=10), plt.xticks([]), plt.yticks([])
            plt.savefig(f'epipolar_{i}_{j}.png')
        valid_feature_mask = np.sum(track_i[i], axis=-1) != -2 #check if the feature is matched in at least one 
        track_i = track_i[:, valid_feature_mask]
        print(f'Found {track_i.shape[1]} features in image {i+1}')
        track = np.concatenate((track, track_i), axis=1)
        
    return track