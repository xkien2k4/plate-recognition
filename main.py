import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from easyocr import Reader
from PIL import Image, ImageDraw, ImageFont, ImageTk


class LicensePlateRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Nhận dạng biển số xe")

        self.file_path = ""

        self.file_button = tk.Button(master, text=" 📂 Nhận dạng từ tệp tin", command=self.select_file)
        self.file_button.pack(pady=10)

        self.webcam_button = tk.Button(master, text=" 📸 BậtBật Camera", command=self.open_webcam)
        self.webcam_button.pack(pady=10)

        self.plate_type = tk.StringVar(value="single")  
        tk.Label(master, text="Chọn kiểu biển số xe:").pack(pady=5)
        self.single_row_radio = tk.Radiobutton(master, text="Hàng 1", variable=self.plate_type, value="single")
        self.single_row_radio.pack()
        self.double_row_radio = tk.Radiobutton(master, text="Hàng 2", variable=self.plate_type, value="double")
        self.double_row_radio.pack()

        self.canvas = tk.Canvas(master, width=400, height=440, bg="white")
        self.canvas.pack(side="left", padx=20, pady=20)

        self.prediction_label = tk.Label(master, text="", font=("Helvetica", 16), wraplength=300, justify="left")
        self.prediction_label.pack(side="right", padx=20, pady=20)

        self.webcam_window_open = False  

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if self.file_path:
            self.perform_image_recognition(cv2.imread(self.file_path))

    def open_webcam(self):
        if self.webcam_window_open:
            return  
        self.webcam_window_open = True
        self.webcam = cv2.VideoCapture(0)

        def show_webcam():
            ret, frame = self.webcam.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                draw = ImageDraw.Draw(frame_pil)
                width, height = frame_pil.size
                draw.line([(0, height // 3), (width, height // 3)], fill="red", width=2)
                draw.line([(0, 2 * height // 3), (width, 2 * height // 3)], fill="red", width=2)
                draw.line([(width // 3, 0), (width // 3, height)], fill="red", width=2)
                draw.line([(2 * width // 3, 0), (2 * width // 3, height)], fill="red", width=2)

                frame_tk = ImageTk.PhotoImage(frame_pil)

                self.webcam_canvas.create_image(0, 0, image=frame_tk, anchor="nw")
                self.webcam_canvas.image = frame_tk  
            if self.webcam_window_open:
                self.webcam_canvas.after(10, show_webcam)  

        self.webcam_window = tk.Toplevel(self.master)
        self.webcam_window.title("Webcam")

        self.webcam_canvas = tk.Canvas(self.webcam_window, width=800, height=600)
        self.webcam_canvas.pack()

        capture_button = tk.Button(self.webcam_window, text="Chụp ảnh", command=self.capture_snapshot)
        capture_button.pack(pady=10)

        self.webcam_window.protocol("WM_DELETE_WINDOW", self.close_webcam_window)

        show_webcam()

    def capture_snapshot(self):
        ret, frame = self.webcam.read()
        if ret:
            self.perform_image_recognition(frame)

    def close_webcam_window(self):
        self.webcam_window_open = False
        self.webcam.release()
        self.webcam_window.destroy()

    def perform_image_recognition(self, img):
        img = cv2.resize(img, (800, 600))
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edged = cv2.Canny(blurred, 10, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        number_plate_shape = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approximation) == 4: 
                number_plate_shape = approximation
                break

        if number_plate_shape is None:
            self.prediction_label.config(text="Không tìm thấy biển số xe.")
            self.display_image(img)
            return

        (x, y, w, h) = cv2.boundingRect(number_plate_shape)
        number_plate = grayscale[y:y + h, x:x + w]

        plate_type = self.plate_type.get()

        reader = Reader(['en', 'vi'], gpu=False)  

        if plate_type == "single":

            text = reader.readtext(number_plate, detail=0)
            if len(text) == 0:
                final_text = "Không tìm thấy biển số xe."
            else:
                final_text = f"Biển số xe: {' '.join(text)}"
        else:
            height, width = number_plate.shape
            top_half = number_plate[0:height // 2, :]  
            bottom_half = number_plate[height // 2:, :]  

            top_text = reader.readtext(top_half, detail=0)
            bottom_text = reader.readtext(bottom_half, detail=0)

            if not top_text and not bottom_text:
                final_text = "Không tìm thấy biển số xe."
            else:
                final_text = f"Biển số xe: {' '.join(top_text)} {' '.join(bottom_text)}"

        cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
        self.prediction_label.config(text=final_text)
        self.display_image(img)

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(200, 220, image=img_tk, anchor="center")
        self.canvas.image = img_tk  


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
