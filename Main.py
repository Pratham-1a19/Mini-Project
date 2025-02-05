try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
    from tkinter import *
    import numpy as np
    import sqlite3
    import webbrowser
    import cv2
    import time
    import os
    from datetime import date
    from PIL import Image, ImageTk

# Database and face cascade
DB_PATH = "customer_db.sqlite"
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class SampleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self._frame = None
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.configure(bg="#69b5b5")
        self._frame.pack()

class StartPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1370x1050+0+0')
        master.configure(bg="#69b5b5")

        def selection():
            a = radio.get()
            if a == 1:
                master.switch_frame(PageOne)
            elif a == 2:
                master.switch_frame(PageTwo)
        
        radio = IntVar()
        tk.Label(self, text="Welcome to the store!!", font=('Poppins', 50), fg="#f8f8f2", bg="#44475a").pack(pady=50)
        tk.Radiobutton(self, text="New User", bg="#6969b5", fg="#e8edf3", variable=radio, value=1, command=selection, width=15, height=1, font=("Poppins", 50)).place(x=100, y=650)
        tk.Radiobutton(self, text="Existing User", bg="#6969b5", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, font=("Poppins", 50)).place(x=775, y=650)

class PageOne(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#b56969")
        tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)

        def training_and_collection():
            cam = cv2.VideoCapture(0)
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            count = 0
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite(f"dataset/User.2.{count}.jpg", gray[y:y+h, x:x+w])
                    cv2.imshow('CollectDatasets', img)
                if cv2.waitKey(1) == ord('q') or count >= 50:
                    break
            cam.release()
            cv2.destroyAllWindows()

            path = 'dataset'
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            def getImagesAndLabels(path):
                image_paths = [os.path.join(path, f) for f in os.listdir(path)]
                face_samples, ids = [], []
                for image_path in image_paths:
                    pil_img = Image.open(image_path).convert('L')
                    img_numpy = np.array(pil_img, 'uint8')
                    id_ = int(os.path.split(image_path)[-1].split(".")[1])
                    faces = detector.detectMultiScale(img_numpy)
                    for (x, y, w, h) in faces:
                        face_samples.append(img_numpy[y:y+h, x:x+w])
                        ids.append(id_)
                return face_samples, ids
            
            faces, ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            recognizer.write('trainer.yml')
            tk.Label(self, text="Datasets are Trained...", fg="#e8edf3", bg="#22264b", width=45, height=1, font=("Poppins", 50)).place(x=0, y=880)
        
        tk.Button(self, text="DATASET COLLECTION", command=training_and_collection, relief=RAISED, width=20, fg="#22264b", bg='#e6cf8b', font=("Poppins", 30)).place_configure(x=950,y=500,anchor="e",height=50,width=500)

class PageTwo(tk.Frame):
    def __init__(self, master):        
        super().__init__(master, bg="#b56969")
        tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=700, pady=700)

        def recognize_user():
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer.yml')
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            names = ['None', 'Pratham', 'Mrudul']
            cam = cv2.VideoCapture(0)
            minW, minH = 0.1 * cam.get(3), 0.1 * cam.get(4)
            i = 0
            
            while i <= 30:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                i += 1
                faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    if 25 <= confidence <= 40:
                        id_text = names[id_]
                    else:
                        id_text = 'unknown'
                    cv2.putText(img, f"{id_text} {100 - confidence:.2f}%", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Recognition', img)
                if cv2.waitKey(1) == ord('q'):
                    break
            cam.release()
            cv2.destroyAllWindows()
            tk.Label(self, text=f"Welcome to the store {id_text}", fg="#e8edf3", bg="#22264b", width=45, height=1, font=("Poppins", 50)).place(x=0, y=5)

        tk.Button(self, text="RECOGNIZE USER", command=recognize_user, relief=RAISED, width=20, bg="#efe9d5", fg="#000000", font=("Poppins", 30)).place_configure(x=950,y=500,anchor="e",height=50,width=500)

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()