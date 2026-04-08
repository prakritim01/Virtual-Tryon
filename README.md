Virtual Try ON : AR Virtual Try-On
Virtual Try ON  is an Augmented Reality (AR) application built with Python that allows users to virtually try on clothing in real-time. By leveraging computer vision and pose estimation, the app dynamically overlays 2D clothing assets onto a live webcam feed, adjusting for the user's posture, size, and movement.

🚀 The Challenge
Online shoppers often face uncertainty because they cannot try on clothes before purchasing, which leads to high return rates for businesses. MaisonMuse addresses this by providing an interactive way to visualize fit and style from home.

🛠️ Core Technologies
OpenCV: Used for capturing live video feed, processing frames, and rendering the final output.

MediaPipe: Google’s framework used to track 33 body landmarks in real-time.

Pillow (PIL): Used for advanced image manipulation, including the rotation and resizing of clothing PNGs.

NumPy: Handles numerical calculations for coordinate geometry and the smoothing algorithm.

📐 How It Works
Landmark Detection: The system tracks four key body landmarks: Left/Right Shoulders (11, 12) and Left/Right Hips (23, 24).

Core Calculations:

Anchor Point: The center of the shoulders serves as the "hang" point for the garment.

Scaling: The width and height are calculated based on the distance between shoulder landmarks and torso height.

Rotation: The software calculates the shoulder angle to ensure the shirt tilts with the user.

Jitter Reduction: To solve "jitter" caused by high-precision sensor data, the app uses a smoothing buffer that averages calculations over the last 8 frames for a stable overlay.

🎮 Controls & Real-Time Tuning
The application features a unique real-time tuning system to allow users to find their perfect personalized fit.

[n]: Switch to the next outfit.

[w / s]: Adjust the vertical (Y-axis) position of the garment.

[a / d]: Adjust the width multiplier.

[+ / -]: Increase or decrease the overall scale.

[q]: Quit the application.

🔮 Future Roadmap
Generative AI: Integrating models like VITON to generate realistic fabric folds and lighting.

3D Assets: Moving from 2D PNGs to 3D models for better depth perception.

Gesture Controls: Using hand tracking to switch outfits with a wave.