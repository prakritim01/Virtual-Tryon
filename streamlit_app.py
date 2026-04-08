import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import math
from collections import deque
import mediapipe as mp
import os
import sys

# --- MEDIAPIPE SETUP ---
try:
    mp_pose = mp.solutions.pose
except ImportError:
    print("Mediapipe not installed. Run: pip install mediapipe")
    sys.exit()

# ----------------------------
# 1. CONSTANTS & CACHED ASSET LOADING (Adapted for Tkinter)
# ----------------------------
CLOTHES_DEFINITION = {
    "grey_tshirt": "assets/grey.png",
    "stripes_jacket": "assets/stripes.png",
    "white_tshirt": "assets/white.png",
    "orange_tshirt": "assets/orange.png",
}

def get_clothing_assets(clothes_dict):
    """Loads all clothing image assets."""
    clothes_pil = {}
    missing_files = []
    for k, v in clothes_dict.items():
        try:
            # Using Image.LANCZOS for good quality loading
            clothes_pil[k] = Image.open(v).convert("RGBA")
        except FileNotFoundError:
            missing_files.append(v)
    
    if missing_files:
        print(f"❌ Missing required clothing files! Please ensure these files exist in the 'assets' folder: {', '.join(missing_files)}")
        # Use a messagebox for the GUI
        messagebox.showerror("Asset Error", f"Missing files in 'assets' folder: {', '.join(missing_files)}")
        sys.exit()
    return clothes_pil

CLOTHES_PIL_CACHE = get_clothing_assets(CLOTHES_DEFINITION)
OUTFIT_NAMES = list(CLOTHES_DEFINITION.keys())


# ----------------------------
# 3. HELPER FUNCTIONS (UNCHANGED)
# ----------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def overlay_pil_on_cv2(bg_bgr, overlay_pil, x, y):
    """Overlays an RGBA PIL image onto a BGR OpenCV frame, handling transparency."""
    overlay_np = np.array(overlay_pil)
    bgr_overlay = cv2.cvtColor(overlay_np[:, :, :3], cv2.COLOR_RGB2BGR) 
    alpha_mask = overlay_np[:, :, 3] / 255.0

    h, w, _ = bg_bgr.shape
    H, W, _ = bgr_overlay.shape
    
    y1, y2 = max(0, int(y)), min(h, int(y) + H)
    x1, x2 = max(0, int(x)), min(w, int(x) + W)
    
    H_actual = y2 - y1
    W_actual = x2 - x1
    
    if H_actual <= 0 or W_actual <= 0:
        return bg_bgr

    bgr_overlay_cropped = bgr_overlay[0:H_actual, 0:W_actual]
    alpha_mask_cropped = alpha_mask[0:H_actual, 0:W_actual][:, :, np.newaxis]
    
    bg_region = bg_bgr[y1:y2, x1:x2]
    
    # Perform alpha blending
    blended_region = (bgr_overlay_cropped.astype(float) * alpha_mask_cropped) + (bg_region.astype(float) * (1 - alpha_mask_cropped))
    
    bg_bgr[y1:y2, x1:x2] = blended_region.astype(bg_bgr.dtype)

    return bg_bgr


# ----------------------------
# 4. AR Try-On Logic (Simplified Transformer Logic)
# ----------------------------
class ARTryOnProcessor:
    def __init__(self, clothes_pil, initial_outfit_key, outfit_names):
        # Initialize Mediapipe Pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
        self.clothes_pil = clothes_pil
        self.outfit_names = outfit_names
        self.selected_outfit = initial_outfit_key

        # Smoothing buffers for stability
        smoothing_window = 8
        self.x_buffer = deque(maxlen=smoothing_window)
        self.y_buffer = deque(maxlen=smoothing_window)
        self.w_buffer = deque(maxlen=smoothing_window)
        self.h_buffer = deque(maxlen=smoothing_window)
        self.angle_buffer = deque(maxlen=smoothing_window)

        # Placement parameters
        self.SCALE_FACTOR = 0.90
        self.Y_OFFSET_PIXELS = 50 
        self.W_MULTIPLIER = 1.15
        self.H_MULTIPLIER = 0.85
        
    def switch_outfit(self):
        """Cycles to the next outfit and returns the new key."""
        current_idx = self.outfit_names.index(self.selected_outfit)
        new_idx = (current_idx + 1) % len(self.outfit_names)
        self.selected_outfit = self.outfit_names[new_idx]
        return self.selected_outfit

    def process_frame(self, frame_bgr):
        """Processes the video frame for AR overlay."""
        frame_out = frame_bgr.copy()
        img_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        
        h, w, _ = frame_out.shape
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 1. Get landmark coordinates
            left_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            right_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            left_hip = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h))
            right_hip = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
            
            # 2. Calculate center points and dimensions
            shoulders_center = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)
            shoulder_width = distance(left_sh, right_sh)
            torso_height = distance(shoulders_center, ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2))
            current_angle = angle_between_points(right_sh, left_sh) 
            
            # 3. Calculate raw clothing dimensions and position
            w_cloth_raw = int(shoulder_width * self.W_MULTIPLIER * self.SCALE_FACTOR)
            h_cloth_raw = int(torso_height * self.H_MULTIPLIER * self.SCALE_FACTOR)
            x_raw = int(shoulders_center[0] - w_cloth_raw / 2)
            y_raw = int(shoulders_center[1] - h_cloth_raw / 2 + self.Y_OFFSET_PIXELS) 

            # 4. Apply smoothing
            self.x_buffer.append(x_raw); self.y_buffer.append(y_raw)
            self.w_buffer.append(w_cloth_raw); self.h_buffer.append(h_cloth_raw)
            self.angle_buffer.append(current_angle)

            x_smooth, y_smooth = int(np.mean(self.x_buffer)), int(np.mean(self.y_buffer))
            w_smooth, h_smooth = int(np.mean(self.w_buffer)), int(np.mean(self.h_buffer))
            angle_smooth = np.mean(self.angle_buffer)

            # 5. Load, rotate, and resize the clothing image
            pil_cloth = self.clothes_pil[self.selected_outfit]
            pil_rotated = pil_cloth.rotate(angle_smooth, resample=Image.BICUBIC, expand=True)

            aspect = pil_rotated.width / pil_rotated.height
            new_w = max(1, w_smooth)
            new_h = max(1, int(new_w / aspect)) 
            
            pil_resized = pil_rotated.resize((new_w, new_h), Image.LANCZOS)
            
            # 6. Calculate final overlay position (centered on the shoulder axis)
            x_center = x_smooth + w_cloth_raw // 2
            y_center = y_smooth + h_cloth_raw // 2

            x_overlay = int(x_center - pil_resized.width // 2)
            y_overlay = int(y_center - pil_resized.height // 2)

            # 7. Overlay the clothing
            frame_out = overlay_pil_on_cv2(frame_out, pil_resized, x_overlay, y_overlay)
        
        return frame_out

    def __del__(self):
        self.pose.close()


# ----------------------------
# 5. TKINTER APP LAYOUT AND MAIN LOOP
# ----------------------------
class ARTryOnApp:
    def __init__(self, master):
        self.master = master
        master.title("Tkinter AR Try-On")

        # 1. Initialize Video Capture
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam. Please ensure it is not in use.")
            sys.exit()

        # Get the resolution from the camera
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 2. Initialize AR Processor
        self.processor = ARTryOnProcessor(CLOTHES_PIL_CACHE, OUTFIT_NAMES[0], OUTFIT_NAMES)

        # 3. Setup Tkinter Widgets
        self.label_vid = tk.Label(master)
        self.label_vid.pack(pady=10)

        self.label_info = tk.Label(master, text=f"Currently Wearing: {self.processor.selected_outfit.replace('_', ' ').title()}", fg="blue")
        self.label_info.pack(pady=5)

        # Button Frame for controls
        control_frame = tk.Frame(master)
        control_frame.pack(pady=10)

        self.btn_switch = tk.Button(control_frame, text="▶️ Next Outfit", command=self.switch_outfit)
        self.btn_switch.pack(side=tk.LEFT, padx=10)
        
        # --- NEW SHUTDOWN BUTTON ---
        self.btn_shutdown = tk.Button(control_frame, text="❌ Exit Application", command=self.shutdown_app, bg="red", fg="white")
        self.btn_shutdown.pack(side=tk.LEFT, padx=10)
        # --- END NEW SHUTDOWN BUTTON ---

        # 4. Start the Video Loop
        self.delay = 15 # Delay in ms, roughly 60 FPS
        self.update_video()

    def switch_outfit(self):
        """Handle outfit switching via button."""
        new_outfit = self.processor.switch_outfit()
        self.label_info.config(text=f"Currently Wearing: {new_outfit.replace('_', ' ').title()}")

    def shutdown_app(self):
        """Closes the main window, triggering cleanup."""
        self.master.destroy()

    def update_video(self):
        """Get a frame from the video source and process it."""
        ret, frame = self.vid.read()
        if ret:
            # Mirror the frame horizontally (common for webcam apps)
            frame = cv2.flip(frame, 1)

            # Process the frame with AR logic
            processed_frame_bgr = self.processor.process_frame(frame)
            
            # Convert OpenCV BGR frame to PIL Image
            img_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

            # Update the Tkinter label with the new image
            self.label_vid.config(image=self.photo)
            self.label_vid.image = self.photo

        # Call the function again after a delay
        self.master.after(self.delay, self.update_video)

    def __del__(self):
        """Release the video source when the application closes."""
        print("Cleaning up resources...")
        if self.vid.isOpened():
            self.vid.release()
            print("Webcam released.")
        del self.processor
        print("MediaPipe processor deleted.")
        cv2.destroyAllWindows()


# --- Main Application Execution ---
if __name__ == "__main__":
    # Ensure 'assets' folder exists for the code to run
    if not os.path.isdir('assets'):
        print("NOTE: Please create an 'assets' folder and place your PNG files inside.")

    root = tk.Tk()
    app = ARTryOnApp(root)
    # The destroy command on the Exit button and the window close 'X' both call __del__ via this protocol binding.
    root.protocol("WM_DELETE_WINDOW", app.shutdown_app) 
    root.mainloop()