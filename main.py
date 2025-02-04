import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from ball_detect_track import BallDetectTrack
from player import Player
from rectify_court import *
from video_handler import VideoHandler
from player_detection import hsv2bgr, COLORS
from tools.plot_tools import plt_plot

TOPCUT = 320  # Global constant for cropping video frames

def get_frames(video_path, central_frame, mod):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in video (from CAP_PROP_FRAME_COUNT):", total_frame_count)
    
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if (index % mod) == 0:
            frames.append(frame[TOPCUT:, :])
        if cv2.waitKey(20) == ord('q'):
            break
        index += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Number of frames extracted: {len(frames)}")
    if len(frames) > central_frame:
        plt.title(f"Central frame shape: {frames[central_frame].shape}")
        plt.imshow(frames[central_frame])
        plt.show()
    else:
        print("Central frame index exceeds extracted frames.")
    
    return frames

if __name__ == '__main__':
    # COURT REAL SIZES: 28m horizontal, 15m vertical
    
    if os.path.exists('resources/pano.png'):
        pano = cv2.imread("resources/pano.png")
    else:
        central_frame = 36
        frames = get_frames('resources/test.mp4', central_frame, mod=3)
        frames_flipped = [cv2.flip(frames[i], 1) for i in range(central_frame)]
        current_mosaic1 = collage(frames[central_frame:], direction=1)
        current_mosaic2 = collage(frames_flipped, direction=-1)
        pano = collage([cv2.flip(current_mosaic2, 1)[:, :-10], current_mosaic1])
        cv2.imwrite("resources/pano.png", pano)
    
    if os.path.exists('resources/pano_enhanced.png'):
        pano_enhanced = cv2.imread("resources/pano_enhanced.png")
        plt_plot(pano, "Panorama")
    else:
        pano_enhanced = pano
        for file in os.listdir("resources/snapshots/"):
            frame = cv2.imread("resources/snapshots/" + file)[TOPCUT:]
            pano_enhanced = add_frame(frame, pano, pano_enhanced, plot=False)
        cv2.imwrite("resources/pano_enhanced.png", pano_enhanced)
    
    pano_enhanced = np.vstack((pano_enhanced,
                               np.zeros((100, pano_enhanced.shape[1], pano_enhanced.shape[2]), dtype=pano.dtype)))
    img = binarize_erode_dilate(pano_enhanced, plot=False)
    simplified_court, corners = rectangularize_court(img, plot=False)
    simplified_court = 255 - np.uint8(simplified_court)
    plt_plot(simplified_court, "Corner Detection", cmap="gray", additional_points=corners)
    rectified = rectify(pano_enhanced, corners, plot=False)
    
    court_map = cv2.imread("resources/2d_map.png")
    scale = rectified.shape[0] / court_map.shape[0]
    court_map = cv2.resize(court_map, (int(scale * court_map.shape[1]), int(scale * court_map.shape[0])))
    resized = cv2.resize(rectified, (court_map.shape[1], court_map.shape[0]))
    court_map = cv2.resize(court_map, (rectified.shape[1], rectified.shape[0]))
    
    video = cv2.VideoCapture("resources/test.mp4")
    total_frames_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in video (CAP_PROP_FRAME_COUNT):", total_frames_video)
    
    players = []
    for i in range(1, 6):
        players.append(Player(i, 'green', hsv2bgr(COLORS['green'][2])))
        players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
    players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))
    
    from player_detection import FeetDetector
    feet_detector = FeetDetector(players)
    ball_detect_track = BallDetectTrack(players)
    video_handler = VideoHandler(pano_enhanced, video, ball_detect_track, feet_detector, court_map)
    video_handler.run_detectors()
