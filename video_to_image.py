import os
import cv2

video_paths = ['sfm_videos/video14.mp4']

# Create the directory for output images, handling any permission issues
output_dir = 'sfm_images'
os.makedirs(output_dir, exist_ok=True)

for video_path in video_paths:
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        continue
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue
    
    idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if idx % 20 == 0:  # Extract every 20th frame
            # Correctly resize frame using OpenCV dimensions (height, width)
            H, W = frame.shape[:2]  # First element is height, second is width
            resized_frame = cv2.resize(frame, (W // 2, H // 2))  # Resize dimensions correctly
            print(W // 2, H // 2)
            # Save the resized frame to a file
            filename = os.path.join(output_dir, f'{idx // 20}.jpg')
            cv2.imwrite(filename, resized_frame)
        idx += 1
