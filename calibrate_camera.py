import cv2 as cv
import numpy as np
import glob

def calibrateCameraFromChessboard(imageDirectory, numCornersX, numCornersY):
    """Performs camera calibration using a directory of chessboard images.

    Args:
        imageDirectory (string): Path pattern like './data/calibration/*.jpeg'.
        numCornersX (int): Number of inner corners in the chessboard's x-direction.
        numCornersY (int): Number of inner corners in the chessboard's y-direction.

    Returns:
        K (list): 3x3 camera intrinsic matrix.
        dist (list): Distortion coefficients.
        newK (list): New intrinsic matrix for undistorted images (use with roi).
        roi (tuple): Region of interest (xi, yi, xf, yf) in undistorted images.
    """
    print('Starting camera calibration...')

    # Define the size of chessboard used
    chessboardSize = (numCornersX, numCornersY)
    terminationCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all images.
    objectPoints = []  # 3d points in real world space
    imagePoints = []  # 2d points in image plane.

    # Prepare object points based on the chessboard size
    objectPoint = np.zeros((1, chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objectPoint[0, :, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Load all images from the directory specified
    images = glob.glob(imageDirectory)
    print('Found', len(images), 'chessboard images.')

    for imageFile in images:
        img = cv.imread(imageFile)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        print('Chessboard detection status:', ret)

        if ret:
            objectPoints.append(objectPoint)
            # Refine the corner positions
            refinedCorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), terminationCriteria)
            imagePoints.append(refinedCorners)
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, chessboardSize, refinedCorners, ret)

    # Perform camera calibration to find the intrinsic matrix, distortion coefficients and more
    calibrationSuccess, intrinsicMatrix, distortionCoefficients, rotationVectors, translationVectors = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None)
    print(intrinsicMatrix)
    return

if __name__ == "__main__":
    calibrateCameraFromChessboard('calibration_images/*.jpg', 10, 7)
