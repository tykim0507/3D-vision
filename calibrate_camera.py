import cv2
import numpy as np
import glob

def calculate_intrinsic_matrix(image_path, grid_x, grid_y):
    """Calculates the intrinsic camera matrix using chessboard images.
    
    Args:
        image_path (str): Path to the folder containing chessboard images (e.g., 'path/to/images/*.jpg')
        grid_x (int): Number of inner corners in the chessboard's x-direction
        grid_y (int): Number of inner corners in the chessboard's y-direction

    Returns:
        K (numpy.ndarray): Intrinsic camera matrix
    """
    print("Starting intrinsic matrix calculation...")
    
    # Define the chessboard size and criteria for corner detection and refinement
    chessboard_size = (grid_x, grid_y)
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points based on the chessboard size
    object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:grid_x, 0:grid_y].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all images
    all_obj_points = []
    all_img_points = []
    
    # Loop over images at the provided path
    for image_file in glob.glob(image_path):
        image = cv2.imread(image_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        success, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
        
        if success:
            # Refine corner position
            corners_refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), termination_criteria)
            all_obj_points.append(object_points)
            all_img_points.append(corners_refined)

    # Perform camera calibration to get only the intrinsic matrix
    _, K, _, _, _ = cv2.calibrateCamera(all_obj_points, all_img_points, gray_image.shape[::-1], None, None)
    
    print("Intrinsic Matrix Calculation Successful.\nIntrinsic Matrix:\n", K)
    
    return K

# Usage example:
if __name__ == '__main__':
    intrinsic_matrix = calculate_intrinsic_matrix('calibration_images/*.jpg', 10, 7)
