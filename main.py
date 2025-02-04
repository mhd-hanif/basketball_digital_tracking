import os
import csv

from matplotlib import pyplot as plt

from ball_detect_track import BallDetectTrack
from player import Player
from rectify_court import *
from video_handler import *


def get_frames(video_path, central_frame, mod):
    frames = []
    cap = cv2.VideoCapture(video_path)
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if (index % mod) == 0:
            frames.append(frame[TOPCUT:, :])

        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break

        if cv2.waitKey(20) == ord('q'): break
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Number of frames : {len(frames)}")
    plt.title(f"Centrale {frames[central_frame].shape}")
    plt.imshow(frames[central_frame])
    plt.show()

    return frames

def save_tracking_data(players, puck_holder_record,
                       offensive_filename="offensive_players.csv",
                       defensive_filename="defensive_players.csv",
                       puck_filename="puck_holder.csv"):
    offensive_rows = []
    defensive_rows = []
    
    # For each player, iterate over the recorded positions.
    for player in players:
        # player.positions is assumed to be a dictionary: frame -> (x, y)
        for frame in sorted(player.positions.keys()):
            pos = player.positions[frame]
            # Skip if position is not available
            if pos is None:
                continue
            row = [frame, player.ID, pos[0], pos[1]]
            # Here we assume players with team "green" are offensive and "white" are defensive.
            if player.team.lower() == "green":
                offensive_rows.append(row)
            elif player.team.lower() == "white":
                defensive_rows.append(row)
            # You can choose to ignore any others (e.g. referee)

    # Write offensive players CSV.
    with open(offensive_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "player_id", "x", "y"])
        writer.writerows(offensive_rows)

    # Write defensive players CSV.
    with open(defensive_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "player_id", "x", "y"])
        writer.writerows(defensive_rows)

    # For puck holder, puck_holder_record is a dictionary: frame -> puck_id
    puck_rows = []
    for frame in sorted(puck_holder_record.keys()):
        puck_id = puck_holder_record[frame]
        puck_rows.append([frame, puck_id])
    with open(puck_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "player_id"])
        writer.writerows(puck_rows)

#####################################################################
if __name__ == '__main__':
    # COURT REAL SIZES
    # 28m horizontal lines
    # 15m vertical lines

    # loading already computed panoramas
    if os.path.exists('resources/pano.png'):
        pano = cv2.imread("resources/pano.png")
    else:
        central_frame = 36
        # frames = get_frames('resources/Short4Mosaicing.mp4', central_frame, mod=3)
        frames = get_frames('resources/test_2.mp4', central_frame, mod=3)
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

    ###################################
    pano_enhanced = np.vstack((pano_enhanced,
                               np.zeros((100, pano_enhanced.shape[1], pano_enhanced.shape[2]), dtype=pano.dtype)))
    img = binarize_erode_dilate(pano_enhanced, plot=False)
    simplified_court, corners = (rectangularize_court(img, plot=False))
    simplified_court = 255 - np.uint8(simplified_court)

    plt_plot(simplified_court, "Corner Detection", cmap="gray", additional_points=corners)

    rectified = rectify(pano_enhanced, corners, plot=False)

    # correspondences map-pano
    map = cv2.imread("resources/2d_map.png")
    scale = rectified.shape[0] / map.shape[0]
    map = cv2.resize(map, (int(scale * map.shape[1]), int(scale * map.shape[0])))
    resized = cv2.resize(rectified, (map.shape[1], map.shape[0]))
    map = cv2.resize(map, (rectified.shape[1], rectified.shape[0]))

    # video = cv2.VideoCapture("resources/Short4Mosaicing.mp4")
    video = cv2.VideoCapture("resources/test_2.mp4")

    players = []
    for i in range(1, 6):
        players.append(Player(i, 'green', hsv2bgr(COLORS['green'][2])))
        players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
    players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))

    feet_detector = FeetDetector(players)
    ball_detect_track = BallDetectTrack(players)
    video_handler = VideoHandler(pano_enhanced, video, ball_detect_track, feet_detector, map)
    video_handler.run_detectors()

    # Now save the tracking data to CSV files.
    # Assume that the players used in tracking are stored in ball_detector.players.
    all_players = video_handler.ball_detector.players
    save_tracking_data(all_players, video_handler.puck_holder_record)
    print("Tracking data saved to CSV files.")
