🚗 Vehicle Speed Estimation using YOLOv8 + ByteTrack

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/5aa6c773-e42e-4323-bb0f-892f98fa044b" />

Accurate real-world vehicle speed estimation from monocular traffic videos using modern computer vision techniques.
This project demonstrates how object detection, multi-object tracking, and perspective geometry can be combined to estimate speed in km/h without radar or LiDAR.

✨ Key Features

YOLOv8 for real-time vehicle detection

ByteTrack for stable multi-object tracking

Perspective (homography) transform to convert pixel motion → meters

Speed estimation in km/h using real-time frame history

Works on high-resolution (4K / ultrawide) videos

Clean visualization with readable speed labels

Fully offline (no cloud dependencies)

<img width="950" height="543" alt="image" src="https://github.com/user-attachments/assets/147e3784-a72f-4b85-ae3b-423e5094e4ab" />


🧠 How It Works (High-Level)

Detect vehicles in each frame using YOLOv8

Track vehicles across frames using ByteTrack (assigns IDs)

Select the bottom-center point of each vehicle

Apply a perspective transform to map the road into a bird’s-eye view

Measure distance traveled (meters) over time (seconds)

Convert to km/h and render on video

📐 Calibration (Most Important Step)

Speed accuracy depends on correct calibration.

A trapezoidal road polygon is selected in the original camera view

This polygon is mapped to a real-world rectangle (meters)

Vertical movement in the transformed view corresponds to real distance

SOURCE points order:
1. Top-left of road
2. Top-right of road
3. Bottom-right of road
4. Bottom-left of road



Adjusting these points adapts the system to any fixed camera.

🗂 Project Structure
vehicle/
├── speed.py              # Main pipeline
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── images/               # Project screenshots (git-tracked)
├── videos/               # Demo clips (git-ignored or Git LFS)
└── .gitignore

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the pipeline
python speed.py

3️⃣ Output

Annotated video saved as output.mp4


📊 Typical Speed Ranges (Sanity Check)

Cars: 30–90 km/h

Bikes: 20–70 km/h

Buses/Trucks: 20–60 km/h

If values fall far outside this range, recalibrate the road polygon.

⚠️ Limitations

Assumes fixed camera

Best for straight road segments

Accuracy depends on correct real-world scale estimation

Not suitable for curved roads without lane-wise calibration

🚀 Future Improvements

Lane-specific calibration

Speed smoothing (EMA / median filtering)

Speed-limit violation detection

CSV export for analytics

Real-time webcam support

👤 Author

Vishal Kanniappan Selvaraj
Computer Science | AI & Data Science
Portfolio experiment for computer vision & intelligent systems

📜 License

This project is intended for educational and research purposes.

