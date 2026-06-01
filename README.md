# 🪞 MaisonMuse: AR Virtual Try-On Studio

An interactive Augmented Reality (AR) application built with Python and Computer Vision that allows users to virtually try on clothing in real-time. By leveraging MediaPipe pose estimation and dynamic matrix math, the engine overlays 2D clothing assets onto a live webcam feed, adjusting perfectly for posture, scale, and movement.

##  The Challenge
Online shoppers often face uncertainty because they cannot physically test clothing before purchasing, leading to high return rates and poor user experience. **MaisonMuse** solves this by providing a high-fidelity, interactive environment to visualize fit and style directly from home.

##  Core Engineering & Architecture
* **Landmark Detection:** Utilizes Google’s **MediaPipe** framework to track 33 sub-millimeter body landmarks in real-time, focusing specifically on the shoulder (11, 12) and hip (23, 24) axes.
* **Dynamic Scaling & Rotation:** The core engine calculates Euclidean distances between anchor points to dynamically scale garment width and height. It continuously computes the shoulder angle using `math.atan2` to ensure the fabric tilts naturally with the user.
* **Jitter Reduction:** High-precision sensor data often causes visual stuttering. MaisonMuse implements a mathematical smoothing buffer (using `collections.deque`) that averages positional matrix calculations over the last 8 frames, resulting in a locked, stable overlay.
* **Image Processing:** Uses **OpenCV** for live video capture/rendering and **Pillow (PIL)** for advanced RGBA alpha-blending and Lanczos resampling of the clothing PNGs. Includes an automated asset pipeline (`process_assets.py`) for stripping background pixels.

##  Tech Stack
* **Language:** Python 3.10+
* **Computer Vision:** OpenCV, MediaPipe
* **Matrix/Math:** NumPy
* **Interface:** Tkinter (Custom Desktop Architecture)

##  How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/prakritim01/virtual-tryon.git](https://github.com/prakritim01/virtual-tryon.git)
cd virtual-tryon
2. Install Dependencies
pip install -r requirements.txt
3. Launch the Application
You can run the full desktop GUI or the lightweight AR engine directly:
# To launch the full MaisonMuse Desktop GUI:
python main.py

# To launch the lightweight OpenCV HUD version:
python ar_tryon_live.py
