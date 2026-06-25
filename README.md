# MaisonMuse | AR Virtual Try-On Studio

An interactive Augmented Reality (AR) application built with Python and Computer Vision. MaisonMuse allows users to virtually try on clothing in real-time by leveraging MediaPipe pose estimation to overlay 2D garment assets onto a live webcam feed.

## Engineering Architecture

- Pose Estimation: Utilizes Google MediaPipe for high-fidelity skeletal landmark tracking.
- Rendering Engine: Employs an OpenCV-based pipeline with PIL-based texture mapping and alpha-channel transparency management.
- Stability: Features a rolling-average temporal smoothing algorithm to eliminate skeletal jitter and ensure fluid overlay attachment.
- Features: Real-time outfit cycling, dynamic Y-axis/width adjustment, and high-resolution snapshot capture.

## How to Run

1. Prerequisites
Ensure you have Python 3.10+ installed.

2. Setup Virtual Environment
Navigate to the project folder and run:
python -m venv venv
(Windows) .\venv\Scripts\activate
(macOS/Linux) source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Process Assets
Ensure your clothing images in the assets/ folder are processed for transparency:
python process_assets.py

5. Launch AR Studio
Launch the lightweight HUD version:
python ar_tryon_live.py

Launch the GUI version:
python main.py

## Controls (Live HUD Version)

- n: Cycle to the next outfit
- c: Capture high-res snapshot
- w / s: Adjust Y-axis position
- a / d: Adjust width scaling
- +/-: Adjust overall scale
- q: Quit application

## Future Roadmap

- Occlusion Handling: Implementation of U-Net/DeepLabV3 segmentation to handle body-arm occlusion (allowing the user to place their arms in front of the garment).
- Advanced Warping: Integration of Thin Plate Spline (TPS) warping for advanced fabric-to-body surface mapping and 3D contour fitting.
