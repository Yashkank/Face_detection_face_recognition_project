# Face_detection_face_recognition_project
This project is a robust, real-time face detection and recognition system designed to automate attendance management using live surveillance footage. Built with InsightFace, OpenCV, and Tkinter, the system integrates an RTSP IP camera feed to detect and recognize human faces efficiently and log attendance securely.The core functionality of the system relies on deep learning-based face analysis, which includes facial detection, alignment, and embedding extraction using InsightFace’s powerful buffalo_s model. Once a face is detected from the video stream, it is compared against pre-registered known face embeddings using cosine similarity to determine identity. The recognition accuracy is optimized with precise threshold tuning, confidence evaluation, and efficient vector matching techniques.
To ensure performance and system efficiency, the system processes every nth frame (as configured), minimizing CPU overload and unnecessary operations. It also logs detection time, recognition time, and CPU performance metrics using the psutil library to monitor resource utilization in real time. Recognized faces are logged into a timestamped CSV file, while unknown faces are ignored or flagged, depending on confidence level. Additionally, repeat logging is controlled to prevent multiple entries for the same person within a short time window.The system features a simple but functional GUI built with Tkinter, allowing the user to start and stop the camera stream with a click. Recognized faces trigger a short audio alert using the winsound module for immediate feedback.

This project emphasizes accuracy, efficiency, and scalability, making it a powerful base for use cases in:
Corporate attendance systems
Campus security
Smart office environments
Event entry management
Surveillance and monitoring systems

Key Features:
Real-time face detection and recognition from RTSP-based IP cameras
Deep learning-powered face analysis using InsightFace (buffalo_s model)
Detailed logging of detection/recognition timestamps, latency, and CPU usage
Automatic attendance CSV generation and controlled logging intervals
Live preview with bounding boxes, identity display, and confidence score
Audio alert for successful recognition events
Multi-threaded video stream handling for smooth frame capture
Tkinter GUI for user-friendly start/stop control
Preprocessing known faces into embeddings with persistent saving/loading
CPU frequency monitoring for system health insights

Technologies Used:
Programming Language: Python
Libraries & Frameworks: OpenCV, InsightFace, NumPy, Scikit-learn, Tkinter, psutil
Streaming Protocol: RTSP (Real Time Streaming Protocol)
Recognition Logic: Cosine similarity over normalized embeddings
Data Storage: CSV logs, Numpy .npy embedding storage



