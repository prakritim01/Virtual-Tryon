import cv2
import numpy as np
from PIL import Image
import math
from collections import deque
import sys 
import os
from datetime import datetime

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
    
    if paste_x + width > 0 and paste_y + height > 0 and paste_x < bg_bgr.shape[1] and paste_y < bg_bgr.shape[0]:
        try:
            bg_pil.paste(overlay_pil, (paste_x, paste_y), overlay_pil)
        except ValueError:
            pass 
    return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def draw_premium_hud(frame, text_lines):
    """Draws a sleek, semi-transparent dark panel behind the UI text."""
    overlay = frame.copy()
    # Draw dark rectangle for HUD background
    cv2.rectangle(overlay, (5, 5), (320, 40 + (len(text_lines) * 25)), (15, 15, 15), -1)
    
    # Alpha blend the overlay with the original frame (Opacity 0.7)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    for idx, line in enumerate(text_lines):
        # Highlight the outfit name in cyan, rest in white
        color = (255, 229, 0) if idx == 0 else (240, 240, 240)
        font_scale = 0.65 if idx == 0 else 0.55
        thickness = 2 if idx == 0 else 1
        
        cv2.putText(frame, line, (15, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
        y_pos += 25
    return frame

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
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.clothes_pil = clothes_pil
        self.outfit_names = list(CLOTHES.keys())
        self.idx = 0
        self.selected_outfit = self.outfit_names[self.idx]
        
        self.smoothing_window = smoothing_window
        self.x_buffer = deque(maxlen=smoothing_window)
        self.y_buffer = deque(maxlen=smoothing_window)
        self.w_buffer = deque(maxlen=smoothing_window)
        self.h_buffer = deque(maxlen=smoothing_window)
        self.angle_buffer = deque(maxlen=smoothing_window)
        
        self.SCALE_FACTOR = 0.90    
        self.Y_OFFSET_PIXELS = 50   
        self.W_MULTIPLIER = 1.15    
        self.H_MULTIPLIER = 0.85    

    def switch_outfit(self):
        self.idx = (self.idx + 1) % len(self.outfit_names)
        self.selected_outfit = self.outfit_names[self.idx]

    def process_frame(self, frame):
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
            
            shoulders_center = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)
            hips_center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
            
            shoulder_width = distance(left_sh, right_sh)
            torso_height = distance(shoulders_center, hips_center)
            current_angle = angle_between_points(right_sh, left_sh)
            
            w_cloth_raw = int(shoulder_width * self.W_MULTIPLIER * self.SCALE_FACTOR)
            h_cloth_raw = int(torso_height * self.H_MULTIPLIER * self.SCALE_FACTOR)
            
            x_raw = int(shoulders_center[0] - w_cloth_raw / 2)
            y_raw = int(shoulders_center[1] - h_cloth_raw / 2 + self.Y_OFFSET_PIXELS) 

            self.x_buffer.append(x_raw); self.y_buffer.append(y_raw)
            self.w_buffer.append(w_cloth_raw); self.h_buffer.append(h_cloth_raw)
            self.angle_buffer.append(current_angle)
            
            x_smooth = int(np.mean(self.x_buffer))
            y_smooth = int(np.mean(self.y_buffer))
            w_smooth = int(np.mean(self.w_buffer))
            h_smooth = int(np.mean(self.h_buffer))
            angle_smooth = np.mean(self.angle_buffer)
            
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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Camera not detected.", file=sys.stderr)
            return

        print("  Starting MaisonMuse AR Studio.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)

            # Define the HUD text
            hud_text = [
                f"LOOK: {self.selected_outfit.replace('_', ' ').upper()}",
                "[n] Next | [c] Capture | [q] Quit",
                f"[w/s] Y-Pos Offset : {self.Y_OFFSET_PIXELS}",
                f"[a/d] Width Adjust : {self.W_MULTIPLIER:.2f}",
                f"[+/-] Scale Adjust : {self.SCALE_FACTOR:.2f}"
            ]
            
            # Apply the sleek overlay
            processed_frame = draw_premium_hud(processed_frame, hud_text)

            cv2.imshow("MaisonMuse AR Studio", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                self.switch_outfit()
            elif key == ord("c"): # NEW: Capture Snapshot
                os.makedirs("snapshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshots/ar_look_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Snapshot saved to: {filename}")
            elif key == ord('w'):
                self.Y_OFFSET_PIXELS -= 5
            elif key == ord('s'):
                self.Y_OFFSET_PIXELS += 5
            elif key == ord('a'):
                self.W_MULTIPLIER -= 0.05
            elif key == ord('d'):
                self.W_MULTIPLIER += 0.05
            elif key == ord('=') or key == ord('+'):
                self.SCALE_FACTOR += 0.05
            elif key == ord('-'):
                self.SCALE_FACTOR -= 0.05

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

if __name__ == "__main__":
    try_on_app = AR_TryOn()
    try_on_app.run()