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
    import tkinter.font as tkFont

DB_PATH = "user.db"
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = 1
id_text = 'unknown'

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
        master.geometry('1727x1050+0+0')
        master.configure(bg="#08D9D6")

        def selection():
            a = radio.get()
            if a == 1:
                master.switch_frame(PageOne)
            elif a == 2:
                master.switch_frame(PageTwo)
        
        radio = IntVar()
        bold_font = tkFont.Font(family='Product Sans', size=65, weight='bold')
        tk.Label(self, text="Welcome to the store!!", font=bold_font, fg="#ffffff", bg="#44475a", width=60).pack(pady=50)
        tk.Radiobutton(self, text="New Customer", bg="#44475a", fg="#e8edf3", variable=radio, value=1, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=200, y=650)
        tk.Radiobutton(self, text="Existing Customer", bg="#44475a", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=1000, y=650)

class PageOne(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((75, 75), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        def training_and_collection():
            cam = cv2.VideoCapture(0)
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            count = 0
            global face_id

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1

                    cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])  # The id to the image is provided here
                    cv2.imshow('CollectDatasets', img)
                if cv2.waitKey(1) == ord('q') or count >= 50:
                    break
            cam.release()
            cv2.destroyAllWindows()

            path = 'dataset'
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            face_id += 1
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
            
                face_id += 1
            
            faces, ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            recognizer.write('trainer.yml')
            tk.Label(self, text="Datasets are Trained...", fg="#e8edf3", bg="#22264b", width=45, height=1, font=("Product Sans", 50)).place(x=0, y=880)
        
        bold_font = tkFont.Font(family='Product Sans', size=65, weight='bold')
        tk.Label(self, text="Click below button for dataset collection", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        tk.Button(self, text="DATASET COLLECTION", command=training_and_collection, relief=RAISED, width=20, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=1100,y=500,anchor="e",height=50,width=500)
        tk.Button(self,image=render,text="BACK",command=lambda: master.switch_frame(StartPage),height=76,width=90).place_configure(x=30,y=50)
        

class PageTwo(tk.Frame):
    def __init__(self, master):        
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        global id_text

        load = Image.open("back-arrow.png")
        load = load.resize((75, 75), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        def recognize_user():
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer.yml')
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            names = ['None', 'Pratham', 'Nishchay', 'Shiv']  # The name is to be added behind this
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
            master.switch_frame(PageThree)
            # tk.Label(self, text=f"Welcome to the store {id_text}", fg="#e8edf3", bg="#22264b", width=45, height=1, font=("Product Sans", 50)).place(x=0, y=5)

        bold_font = tkFont.Font(family='Product Sans', size=65, weight='bold')
        tk.Label(self, text="Recognize existing customer", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        tk.Button(self, text="RECOGNIZE USER", command=recognize_user, relief=RAISED, width=20, bg="#efe9d5", fg="#000000", font=("Product Sans", 30)).place_configure(x=1100,y=500,anchor="e",height=50,width=500)
        tk.Button(self,image=render,text="BACK",command=lambda: master.switch_frame(StartPage),height=76,width=90).place_configure(x=30,y=50)

class PageThree(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        global id_text

        load = Image.open("back-arrow.png")
        load = load.resize((75, 75), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=65, weight='bold')
        tk.Label(self, text=f"Welcome {id_text}", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        tk.Button(self,image=render,text="BACK",command=lambda: master.switch_frame(PageTwo),height=76,width=90).place_configure(x=30,y=50)
        

# if __name__ == "__main__":
app = SampleApp()
app.mainloop()