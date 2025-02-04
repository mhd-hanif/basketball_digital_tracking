from player_detection import *
import skvideo.io
import cv2
import csv

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

class VideoHandler:
    def __init__(self, pano, video, ball_detector, feet_detector, map_2d):
        self.M1 = np.load("Rectify1.npy")
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.pano = pano
        self.video = video
        self.kp1, self.des1 = self.sift.compute(pano, self.sift.detect(pano))
        self.feet_detector = feet_detector
        self.ball_detector = ball_detector
        self.map_2d = map_2d
        # Using "ball_holder_record" instead of "puck" (dictionary: frame -> player_id or None)
        self.ball_holder_record = {}
        # Record every frame’s player position as a tuple:
        # (timeframe, player_id, team, x_pixel, y_pixel)
        self.all_player_positions = []
    
    def run_detectors(self):
        writer = skvideo.io.FFmpegWriter("demo2.mp4")
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        time_index = 0
        
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            
            # Crop the frame
            frame = frame[TOPCUT:, :]
            M = self.get_homography(frame, self.des1, self.kp1)
            # get_players_pos updates each player's positions and draws on the 2D map
            frame, self.map_2d, map_2d_text = self.feet_detector.get_players_pos(
                M, self.M1, frame, time_index, self.map_2d)
            frame, ball_map_2d = self.ball_detector.ball_tracker(
                M, self.M1, frame, self.map_2d.copy(), map_2d_text, time_index)
            vis = np.vstack((frame, cv2.resize(map_2d_text, (frame.shape[1], frame.shape[1] // 2))))
            
            # Record the ball holder for this frame.
            ball_id = None
            for p in self.ball_detector.players:
                if p.has_ball:
                    ball_id = p.ID
                    break
            self.ball_holder_record[time_index] = ball_id
            
            # Record player positions for this frame from the feet detector.
            # (Each player object has a .team attribute and their positions stored in .positions.)
            for p in self.feet_detector.players:
                pos = p.positions.get(time_index)
                if pos is not None:
                    self.all_player_positions.append((time_index, p.ID, p.team.lower(), pos[0], pos[1]))
            
            cv2.imshow("Tracking", vis)
            writer.writeFrame(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            
            k = cv2.waitKey(1) & 0xff
            if k == 27:  # exit on 'Esc' key
                break
            
            time_index += 1
            if time_index % 50 == 0:
                print(f"Processed frame {time_index} / {total_frames}")
        
        self.video.release()
        try:
            writer.close()
        except AttributeError:
            pass
        cv2.destroyAllWindows()
        
        # Now, save all tracking data (players and ball holder) into CSV files.
        self.save_tracking_data()
    
    def get_homography(self, frame, des1, kp1):
        kp2 = self.sift.detect(frame)
        kp2, des2 = self.sift.compute(frame, kp2)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return M
        
    def save_tracking_data(self):
        """
        Filter the recorded per-frame player positions by team and convert pixel coordinates to meters.
        Then save:
          - Offensive players (team "green") to "offensive_players_basket.csv"
          - Defensive players (team "white") to "defensive_players_basket.csv"
          - Ball holder intervals (grouped contiguous frames) to "ball_holder_basketball.csv"
        
        The conversion uses a 28 m × 15 m court and the dimensions of self.map_2d.
        """
        # Get dimensions of the 2D court overlay.
        map_height, map_width = self.map_2d.shape[:2]
        court_width_m = 28.0   # horizontal length in meters
        court_height_m = 15.0  # vertical length in meters
        scale_x = court_width_m / map_width
        scale_y = court_height_m / map_height
        
        offensive_data = []
        defensive_data = []
        
        for record in self.all_player_positions:
            frame, player_id, team, x_pixel, y_pixel = record
            # Convert from pixel to meter.
            x_m = x_pixel * scale_x
            y_m = y_pixel * scale_y
            if team == "green":
                offensive_data.append((frame, player_id, x_m, y_m))
            elif team == "white":
                defensive_data.append((frame, player_id, x_m, y_m))
        
        with open("offensive_players_basketball.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timeframe", "player_id", "x", "y"])
            writer.writerows(offensive_data)
        
        with open("defensive_players_basketball.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timeframe", "player_id", "x", "y"])
            writer.writerows(defensive_data)
        
        # Process ball holder record into contiguous intervals.
        ball_holder_intervals = []
        sorted_frames = sorted(self.ball_holder_record.keys())
        if sorted_frames:
            current_start = sorted_frames[0]
            current_player = self.ball_holder_record[current_start]
            previous_frame = current_start
            for frame in sorted_frames[1:]:
                if self.ball_holder_record[frame] == current_player and frame == previous_frame + 1:
                    previous_frame = frame
                else:
                    if current_player is not None:
                        ball_holder_intervals.append((current_start, previous_frame, current_player))
                    current_start = frame
                    current_player = self.ball_holder_record[frame]
                    previous_frame = frame
            if current_player is not None:
                ball_holder_intervals.append((current_start, previous_frame, current_player))
        
        with open("ball_holder_basketball.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Start", "End", "Player"])
            writer.writerows(ball_holder_intervals)
        
        print("Tracking data saved to CSV files.")
