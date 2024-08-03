import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

class VideoCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Capture with PySide6")
        self.setGeometry(100, 100, 800, 600)

        # Create a widget for the main window
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create a label to display the video feed
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Create buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.layout.addWidget(self.stop_button)

        # self.snapshot_button = QPushButton("Capture Snapshot")
        # self.snapshot_button.clicked.connect(self.capture_snapshot)
        # self.layout.addWidget(self.snapshot_button)

        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Additional attributes
        self.is_drawing = False
        self.previous_point = None
        self.drawing_frame = None

        # Key state tracking
        self.key_pressed = False
        self.drawing_color = (0, 255, 0)
        self.last_point = None
        self.thickness = None

    def start_video(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_video(self):
        self.timer.stop()
        self.cap.release()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.key_pressed = True
            self.drawing_color = (0, 255, 0)
            self.thickness = 3


        elif event.key() == Qt.Key_Alt:
            self.key_pressed = True
            self.drawing_color = (0, 0, 0)
            self.thickness = 10

        elif event.key() == Qt.Key_Escape:
            self.drawing_frame = np.zeros_like(self.drawing_frame)

        elif event.key() == Qt.Key_Enter:
            self.capture_snapshot()


    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.key_pressed = False
        elif event.key() == Qt.Key_Alt:
            self.key_pressed = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process frame to detect orange object and draw
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower_color = np.array([0, 61, 170])
        upper_color = np.array([36, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0 and self.key_pressed:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            center = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
            self.last_point = center
            if self.previous_point is None:
                self.previous_point = center
            if self.drawing_frame is None:
                self.drawing_frame = np.zeros_like(frame)
            cv2.line(self.drawing_frame, self.previous_point, center, self.drawing_color, self.thickness)
            self.previous_point = center
        else:
            self.previous_point = None

        # Combine original frame with drawing frame
        if self.drawing_frame is not None:
            combined_frame = cv2.addWeighted(frame, 1, self.drawing_frame, 1, 0)
        else:
            combined_frame = frame

        # Convert combined frame to QImage
        h, w, ch = combined_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(combined_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def capture_snapshot(self):
        # Process the snapshot here
        x = cv2.cvtColor(self.drawing_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(x)
        response = self.get_gemini_response(pil_image)
        draw = ImageDraw.Draw(pil_image)

        # Drawing the text on the image
        draw.text(xy=(self.last_point[0] + 10, self.last_point[1] - 40),
                text=response,
                font=ImageFont.truetype("GeistMono.ttf", 40),
                fill=(0, 255, 0))



        # self.drawing_frame = cv2.putText(self.drawing_frame, response, (self.last_point[0] + 10, self.last_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        self.drawing_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # print(f"Response from Gemini Model: {response}")

    def get_gemini_response(self, image):
        # Function to send image to Gemini model and get response
        # Placeholder for actual implementation

        response = model.generate_content(
            [
                "Can you answer this math question? Please only give answer no other explanation.",
                image,
            ]
        )
        print(response.text)
        return response.text

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoCaptureApp()
    window.show()
    sys.exit(app.exec())
