import cv2
import numpy as np
from PIL import Image
import math
from collections import deque
import sys # Import sys for a clean exit

# Try importing mediapipe, otherwise show clear guidance
try:
    import mediapipe as mp
except ImportError:
    print(
        "  Mediapipe is not installed.\n"
        "Install it with: pip install mediapipe==0.10.8",
        file=sys.stderr
    )
    sys.exit(1)

mp_pose = mp.solutions.pose

# ----------------------------
# Helper Functions
# ----------------------------
def overlay_pil_on_cv2(bg_bgr, overlay_pil, x, y):
    """Overlay a PIL RGBA image onto OpenCV BGR background at (x, y) top-left."""
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    bg_pil = Image.fromarray(bg_rgb)
    paste_x = int(x)
    paste_y = int(y)
    width, height = overlay_pil.size
    # Check bounds before pasting
    if paste_x + width > 0 and paste_y + height > 0 and paste_x < bg_bgr.shape[1] and paste_y < bg_bgr.shape[0]:
        try:
            bg_pil.paste(overlay_pil, (paste_x, paste_y), overlay_pil)
        except ValueError:
            pass # Ignore errors from partial pastes
    return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

def distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between_points(p1, p2):
    """Calculates the angle in degrees between p1 and p2 (shoulder line)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

# ----------------------------
# Clothing Assets & Preload
# ----------------------------
CLOTHES = {
    "grey_tshirt": "assets/grey.png",
    "stripes_jacket": "assets/stripes.png",
    "white_tshirt": "assets/white.png",
    "orange_tshirt": "assets/orange.png",
}

try:
    # Preload clothing images
    clothes_pil = {k: Image.open(v).convert("RGBA") for k, v in CLOTHES.items()}
except FileNotFoundError as e:
    print(
        f"\n  Missing clothing file!\n{e}\n"
        "Please make sure your 'assets' folder is correct and contains all PNGs.",
        file=sys.stderr
    )
    sys.exit(1)

# ----------------------------
# Main AR Try-On Class
# ----------------------------
class AR_TryOn:
    def __init__(self, smoothing_window=8):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.clothes_pil = clothes_pil
        self.outfit_names = list(CLOTHES.keys())
        self.idx = 0
        self.selected_outfit = self.outfit_names[self.idx]
        
        # Smoothing buffers
        self.smoothing_window = smoothing_window
        self.x_buffer = deque(maxlen=smoothing_window)
        self.y_buffer = deque(maxlen=smoothing_window)
        self.w_buffer = deque(maxlen=smoothing_window)
        self.h_buffer = deque(maxlen=smoothing_window)
        self.angle_buffer = deque(maxlen=smoothing_window)
        
        # --- TUNING PARAMETERS (now class variables) ---
        self.SCALE_FACTOR = 0.90    # Overall size
        self.Y_OFFSET_PIXELS = 50   # Vertical position
        self.W_MULTIPLIER = 1.15    # Width (relative to shoulders)
        self.H_MULTIPLIER = 0.85    # Height (relative to torso)

    def switch_outfit(self):
        """Switches to the next outfit."""
        self.idx = (self.idx + 1) % len(self.outfit_names)
        self.selected_outfit = self.outfit_names[self.idx]

    def process_frame(self, frame):
        """Detects pose, calculates overlay parameters, and applies clothing."""
        h, w, _ = frame.shape
        frame_out = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_sh = (int(lm[11].x * w), int(lm[11].y * h))
            right_sh = (int(lm[12].x * w), int(lm[12].y * h))
            left_hip = (int(lm[23].x * w), int(lm[23].y * h))
            right_hip = (int(lm[24].x * w), int(lm[24].y * h))
            
            shoulders_center = (
                (left_sh[0] + right_sh[0]) // 2,
                (left_sh[1] + right_sh[1]) // 2
            )
            hips_center = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2
            )
            
            shoulder_width = distance(left_sh, right_sh)
            torso_height = distance(shoulders_center, hips_center)
            current_angle = angle_between_points(right_sh, left_sh)
            
            # Use the class's tuning parameters
            w_cloth_raw = int(shoulder_width * self.W_MULTIPLIER * self.SCALE_FACTOR)
            h_cloth_raw = int(torso_height * self.H_MULTIPLIER * self.SCALE_FACTOR)
            
            x_raw = int(shoulders_center[0] - w_cloth_raw / 2)
            y_raw = int(shoulders_center[1] - h_cloth_raw / 2 + self.Y_OFFSET_PIXELS) # Use Y_OFFSET

            # Smoothing
            self.x_buffer.append(x_raw); self.y_buffer.append(y_raw)
            self.w_buffer.append(w_cloth_raw); self.h_buffer.append(h_cloth_raw)
            self.angle_buffer.append(current_angle)
            
            x_smooth = int(np.mean(self.x_buffer))
            y_smooth = int(np.mean(self.y_buffer))
            w_smooth = int(np.mean(self.w_buffer))
            h_smooth = int(np.mean(self.h_buffer))
            angle_smooth = np.mean(self.angle_buffer)
            
            # Rotation and Resize
            pil_cloth = self.clothes_pil[self.selected_outfit]
            pil_rotated = pil_cloth.rotate(angle_smooth, resample=Image.BICUBIC, expand=True)
            
            aspect = pil_rotated.width / pil_rotated.height
            new_w = max(1, w_smooth)
            new_h = max(1, int(new_w / aspect))
            
            pil_resized = pil_rotated.resize((new_w, new_h), Image.LANCZOS)
            
            center_x_offset = pil_resized.width // 2
            center_y_offset = pil_resized.height // 2
            
            x_overlay = int(x_smooth - center_x_offset + w_smooth / 2)
            y_overlay = int(y_smooth - center_y_offset + h_smooth / 2)
            
            frame_out = overlay_pil_on_cv2(frame_out, pil_resized, x_overlay, y_overlay)
            
        return frame_out

    def run(self):
        """Main loop for video capture and processing."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Camera not detected.", file=sys.stderr)
            return

        print("  Starting AR Try-On (Direct Placement).")
        print("  Controls:")
        print("    [n] = Next Outfit")
        print("    [w/s] = Move Up / Down")
        print("    [a/d] = Adjust Width")
        print("    [+/-] = Adjust Scale")
        print("    [q] = Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Could not read frame.", file=sys.stderr)
                break
                
            processed_frame = self.process_frame(frame)

            # --- NEW: Updated UI Text ---
            ui_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Line 1: Outfit
            cv2.putText(processed_frame, f"Outfit: {self.selected_outfit.replace('_', ' ').title()}", 
                        (10, 30), font, 0.7, ui_color, 2, cv2.LINE_AA)
            # Line 2: Controls
            cv2.putText(processed_frame, "[n] Next [q] Quit", 
                        (10, 60), font, 0.7, ui_color, 2, cv2.LINE_AA)
            # Line 3: Tuning Info (Y-Offset)
            cv2.putText(processed_frame, f"[w/s] Y-Pos: {self.Y_OFFSET_PIXELS}", 
                        (10, 90), font, 0.6, ui_color, 1, cv2.LINE_AA)
            # Line 4: Tuning Info (Width)
            cv2.putText(processed_frame, f"[a/d] Width: {self.W_MULTIPLIER:.2f}", 
                        (10, 115), font, 0.6, ui_color, 1, cv2.LINE_AA)
            # Line 5: Tuning Info (Scale)
            cv2.putText(processed_frame, f"[+/-] Scale: {self.SCALE_FACTOR:.2f}", 
                        (10, 140), font, 0.6, ui_color, 1, cv2.LINE_AA)

            cv2.imshow("  AR Try-On - Direct Chest Placement", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                self.switch_outfit()
            
            # --- NEW: Real-time Fit Tuning Controls ---
            elif key == ord('w'): # Move Up
                self.Y_OFFSET_PIXELS -= 5
            elif key == ord('s'): # Move Down
                self.Y_OFFSET_PIXELS += 5
            elif key == ord('a'): # Decrease Width
                self.W_MULTIPLIER -= 0.05
            elif key == ord('d'): # Increase Width
                self.W_MULTIPLIER += 0.05
            elif key == ord('=') or key == ord('+'): # Increase Scale
                self.SCALE_FACTOR += 0.05
            elif key == ord('-'): # Decrease Scale
                self.SCALE_FACTOR -= 0.05

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

if __name__ == "__main__":
    try_on_app = AR_TryOn()
    try_on_app.run()