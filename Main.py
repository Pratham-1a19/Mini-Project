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
    from tkcalendar import DateEntry
    import json
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

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
        master.configure(bg="#FFF1D5")

        def selection():
            a = radio.get()
            if a == 1:
                master.switch_frame(PageSeven)
            elif a == 2:
                master.switch_frame(PageTen)
            elif a == 3: 
                master.switch_frame(PageOne)
        
        radio = IntVar()
        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="General Store", font=bold_font, fg="#ffffff", bg="#44475a", width=60).pack(pady=50)
        tk.Radiobutton(self, text="Admin", bg="#44475a", fg="#e8edf3", variable=radio, value=1, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=125, y=650)
        tk.Radiobutton(self, text="Staff", bg="#44475a", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=875, y=650)

class PageOne(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#08D9D6")

        def selection():
            a = radio.get()
            if a == 1:
                master.switch_frame(PageTwo)
            elif a == 2:
                master.switch_frame(PageThree)
        
        radio = IntVar()
        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="General Store", font=bold_font, fg="#ffffff", bg="#44475a", width=60).pack(pady=50)
        tk.Radiobutton(self, text="New Customer", bg="#44475a", fg="#e8edf3", variable=radio, value=1, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=150, y=650)
        tk.Radiobutton(self, text="Existing Customer", bg="#44475a", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=950, y=650)
     

class PageTwo(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        def training_and_collection():
            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(customer_id) FROM customers")
                result = cursor.fetchone()
                if result and result[0]:
                    face_id = result[0]
                else:
                    tk.Label(self, text="No customer found in the database!", fg="red", bg="#69b5b5", font=("Product Sans", 30)).place(x=0, y=880)
                    return
                conn.close()

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

                        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
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
                tk.Label(self, text="Datasets are Trained...", fg="#e8edf3", bg="#22264b", width=45, height=1, font=("Product Sans", 50)).place(x=0, y=880)
            except Exception as e:
                print(f"Error: {e}")

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Click below button for dataset collection", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        tk.Button(self, text="DATASET COLLECTION", command=training_and_collection, relief=RAISED, width=20, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=1100, y=500, anchor="e", height=50, width=500)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageFive), height=86, width=90).place_configure(x=30, y=50)
        
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

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Recognize Customer", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        tk.Button(self, text="RECOGNIZE USER", command=recognize_user, relief=RAISED, width=20, bg="#efe9d5", fg="#000000", font=("Product Sans", 30)).place_configure(x=1100,y=500,anchor="e",height=50,width=500)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageTen), height=86, width=90).place_configure(x=30, y=50)

class PageFour(tk.Frame):
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

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text=f"Welcome {id_text}", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        tk.Button(self,image=render,text="BACK",command=lambda: master.switch_frame(PageTwo),height=76,width=90).place_configure(x=30,y=50)
        
class PageFive(tk.Frame):
        def __init__(self, master):    
            super().__init__(master)
            self.pack(fill="both", expand=True)
            master.geometry('1727x1050+0+0')
            master.configure(bg="#69b5b5")

            bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
            normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")
            tk.Label(self, text=f"Login Page", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
            
            def callback4(id, pwd):
                # This is a function block
                print(id + " " + pwd)

            tk.Label(self, text="Enter Staff Id:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45,y=300)
            e1 = tk.Entry(master,width=20,font=(normal_font), fg="#22264b", bg="#e8edf3")
            e1.pack()
            e1.place(x = 685, y = 298)
            tk.Label(self, text="Enter Password:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45,y=400)
            e2 = tk.Entry(master,width=20,font=(normal_font),show="*", fg="#22264b", bg="#e8edf3")
            e2.pack()
            e2.place(x = 685, y = 398)
            tk.Button(self, text="Login", command=lambda: callback4(e1.get(),e2.get()),relief=RAISED,width=19,fg="#22264b",bg='#e6cf8b',font=normal_font).place_configure(x=45,y=500)

class PageSix(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")
        tk.Label(self, text="Billing Page", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        
        self.checkboxes = {}
        self.entries = {}
        y_axis = 300

        texts = ["Milk", "Curd", "Buttermilk"]
        for text in texts:
            var = tk.BooleanVar()
            tk.Checkbutton(self, text=text, bg="#44475a", fg="#e8edf3", variable=var, width=10, height=1, bd=5, relief='raised', font=normal_font, highlightbackground="#000000", highlightcolor="#000000").place(x=200, y=y_axis)
            e1 = tk.Entry(master, width=2, font=normal_font)
            e1.pack()
            e1.place(x=550, y=y_axis)
            y_axis += 100
            self.checkboxes[text] = var
            self.entries[text] = e1

        tk.Button(self, text="Submit", command=self.show_selection, relief=RAISED, width=19, fg="#22264b", bg='#e6cf8b', font=normal_font).place_configure(x=200, y=y_axis)
        tk.Button(self,image=render,text="BACK",command=lambda: master.switch_frame(PageFive),height=86,width=90).place_configure(x=30,y=50)

    def show_selection(self):
        selected_items = {text: self.entries[text].get() for text, var in self.checkboxes.items() if var.get()}
        print("Selected items with quantities:", selected_items)

class PageSeven(tk.Frame):
    def __init__(self, master):    
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")
        err_font = tkFont.Font(family="Product Sans", size=25, weight="normal")
        tk.Label(self, text=f"Login Page", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        
        def callback4(id, pwd):
            if id != "admin":
                error_label.config(text="*admin id is incorrect", fg="red")
                
            elif pwd != "admin123":
                error_label.config(text="*password is incorrect", fg="red")

            else:
                master.switch_frame(PageEight)

        tk.Label(self, text="Enter Admin Id:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=300)
        e1 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e1.pack()
        e1.place(x=685, y=298)
        tk.Label(self, text="Enter Password:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=400)
        e2 = tk.Entry(master, width=20, font=(normal_font), show="*", fg="#22264b", bg="#e8edf3")
        e2.pack()
        e2.place(x=685, y=398)
        tk.Button(self, text="Login", command=lambda: callback4(e1.get(), e2.get()), relief=RAISED, width=19, fg="#22264b", bg='#e6cf8b', font=normal_font).place_configure(x=45, y=500)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(StartPage), height=86, width=90).place_configure(x=30, y=50)

        error_label = tk.Label(self, text="", fg="red", bg="#69b5b5", font=err_font)
        error_label.place(x=45, y=600)

class PageEight(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Welcome to the Admin Page", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(StartPage), height=86, width=90).place_configure(x=30, y=50)

        def selection():
            a = radio.get()
            if a == 1:
                master.switch_frame(PageNine)
        
        radio = IntVar()
        tk.Radiobutton(self, text="Add New Staff Member", bg="#44475a", fg="#e8edf3", variable=radio, value=1, command=selection, width=18, height=1, bd=5, relief='raised', font=('Product Sans', 35), highlightbackground="#000000", highlightcolor="#000000").place(x=45, y=250)

class PageNine(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")
        normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Add Staff Member", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        
        tk.Label(self, text="Staff Name:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=300)
        e1 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e1.pack()
        e1.place(x=685, y=298)

        tk.Label(self, text="Gender:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=400)
        e2 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e2.pack()
        e2.place(x=685, y=398)

        tk.Label(self, text="Date of birth:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=500)
        e3 = DateEntry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3", date_pattern="yyyy-mm-dd")
        e3.pack()
        e3.place(x=685, y=498)

        tk.Label(self, text="Email:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=600)
        e4 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e4.pack()
        e4.place(x=685, y=598)

        tk.Label(self, text="Phone number:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=700)
        e5 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e5.pack()
        e5.place(x=685, y=698)

        def add_staff_to_db():
            staff_name = e1.get()
            staff_gender = e2.get()
            staff_dob = e3.get()
            staff_email = e4.get()
            staff_phone = e5.get()

            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()

                cursor.execute("SELECT MAX(staff_id) FROM staff_members")
                result = cursor.fetchone()
                next_id = (result[0] or 0) + 1

                staff_id = f"staff{next_id}"

                cursor.execute("""
                    INSERT INTO staff_members (staff_id, staff_name, staff_gender, staff_dob, staff_email, staff_phone)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (staff_id, staff_name, staff_gender, staff_dob, staff_email, staff_phone))
                conn.commit()
                conn.close()
                tk.Label(self, text="Staff added successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=45, y=900)
            except Exception as e:
                tk.Label(self, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=60, y=850)

        tk.Button(self, text="Add Staff", command=lambda: add_staff_to_db(), relief=RAISED, width=19, fg="#22264b", bg='#e6cf8b', font=normal_font).place_configure(x=45, y=800)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageEight), height=86, width=90).place_configure(x=30, y=50)

class PageTen(tk.Frame):
    def __init__(self, master):    
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")
        err_font = tkFont.Font(family="Product Sans", size=25, weight="normal")
        tk.Label(self, text=f"Staff Login Page", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        
        def callback4(id, pwd):
            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM staff_members WHERE staff_id = ? AND ? = 'staff'", (id, pwd))
                result = cursor.fetchone()

                if result and pwd == "staff":
                    master.switch_frame(PageEleven)
                else:
                    error_label.config(text="*invalid staff id or password", fg="red")

                conn.close()
            except Exception as e:
                error_label.config(text=f"*Error: {e}", fg="red")

        tk.Label(self, text="Enter Staff Id:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=300)
        e1 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e1.pack()
        e1.place(x=685, y=298)
        tk.Label(self, text="Enter Password:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=400)
        e2 = tk.Entry(master, width=20, font=(normal_font), show="*", fg="#22264b", bg="#e8edf3")
        e2.pack()
        e2.place(x=685, y=398)
        tk.Button(self, text="Login", command=lambda: callback4(e1.get(), e2.get()), relief=RAISED, width=19, fg="#22264b", bg='#e6cf8b', font=normal_font).place_configure(x=45, y=500)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(StartPage), height=86, width=90).place_configure(x=30, y=50)

        error_label = tk.Label(self, text="", fg="red", bg="#69b5b5", font=err_font)
        error_label.place(x=45, y=600)

class PageEleven(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Welcome to the Staff Page", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        

        def selection():
            a = radio.get()
            if a == 1:
                master.switch_frame(PageTwelve)
            elif a == 2:
                master.switch_frame(PageThirteen)
            elif a == 3:
                master.switch_frame(PageThree)
        
        radio = IntVar()
        tk.Radiobutton(self, text="Add items", bg="#44475a", fg="#e8edf3", variable=radio, value=1, command=selection, width=10, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=125, y=450)
        tk.Radiobutton(self, text="New Customer", bg="#44475a", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=125, y=550)
        tk.Radiobutton(self, text="Existing Customer", bg="#44475a", fg="#e8edf3", variable=radio, value=3, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=125, y=650)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageTen), height=86, width=90).place_configure(x=30, y=50)

class PageTwelve(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        normal_font = tkFont.Font(family='Product Sans', size=30, weight='normal')
        tk.Label(self, text="Add item to the store", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)
        
        OptionList = ["Please Select Category","Baby Products","Bakery Products","Dairy Products","Fresh Products","Frozen Food","Packaged Products", "Personal Care", "Pet Supplies"]
        variable = StringVar(self)
        variable.set(OptionList[0])

        def callback9():
            selected_category = variable.get()
            globals()['sub'] = selected_category

            product_name = e1.get()
            product_price = e2.get()
            product_quantity = e3.get()

            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()

                table_name = selected_category
                cursor.execute(f"""
                    INSERT INTO "{table_name}" (product_name, product_qty, product_prize)
                    VALUES (?, ?, ?)
                """, (product_name, product_quantity, product_price))
                conn.commit()
                tk.Label(self, text="Item added successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=45, y=800)

                print(product_name, product_quantity, product_price)

                cursor.execute("SELECT customer_email, customer_name, customer_preference FROM customers")
                customers = cursor.fetchall()

                matching_customers = []
                for email, name, preferences in customers:
                    try:
                        preference_list = json.loads(preferences)
                        if selected_category in preference_list:
                            matching_customers.append((email, name))
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in preferences for customer {name}: {preferences}")

                for email, name in matching_customers:
                    send_email(email, name, selected_category, product_name, product_price)

                conn.close()
            except Exception as e:
                print(f"Error: {e}")
            tk.Label(self, text="Item added successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=45, y=800)
            

        def send_email(email, name, category, product_name, product_price):
            try:
                SMTP_SERVER = "smtp.ethereal.email"
                sender_email = "samanta.gerlach95@ethereal.email"
                sender_password = "4BVcXp5yhjuqm2D8zd"
                subject = "New Item Added to Your Preferred Category"
                body = (
                    f"Respected {name},\n\n"
                    f"We are excited to inform you that a new item, '{product_name}', priced at Rs. {product_price}, "
                    f"has been added to the '{category}' category in our store. Visit us to check it out!\n\n"
                    f"Best regards,\nYour Store Team"
                )

                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = email
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))

                with smtplib.SMTP(SMTP_SERVER, 587) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, email, msg.as_string())

                print(f"Email sent to {email}")
            except Exception as e:
                print(f"Failed to send email to {email}: {e}")


        opt = tk.OptionMenu(master, variable,*OptionList)
        opt.config(width=15,font=("Product Sans", 30))
        tk.Label(self, text="Select product category:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=300)
        opt.pack(side="top",pady=20)
        opt.place(x = 685, y = 298)

        tk.Label(self, text="Enter product name:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=400)
        e1 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e1.pack()
        e1.place(x=685, y=398)

        tk.Label(self, text="Enter product prize:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=500)
        e2 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e2.pack()
        e2.place(x=685, y=498)

        tk.Label(self, text="Enter product quantity:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=600)
        e3 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e3.pack()
        e3.place(x=685, y=598)

        tk.Button(self, text="Add item", command=callback9, relief=RAISED, width=7, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=45,y=700)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageEleven), height=86, width=90).place_configure(x=30, y=50)

class PageThirteen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render
        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")
        
        def training_and_collection():
            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(customer_id) FROM customers")
                result = cursor.fetchone()
                if result and result[0]:
                    face_id = result[0] + 1
                else:
                    tk.Label(self, text="No customer found in the database!", fg="red", bg="#69b5b5", font=("Product Sans", 30)).place(x=0, y=880)
                    return
                conn.close()

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

                        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
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
            except Exception as e:
                print(f"Error: {e}")
            tk.Label(self, text="Dataset trained successfully!", fg="green", bg="#69b5b5", font=("Product Sans", 30)).place(x=550, y=800)

        tk.Label(self, text="New Customer Details", font=bold_font, fg="#f8f8f2", bg="#44475a", width=60).pack(pady=50)

        tk.Label(self, text="Enter name:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=300)
        e1 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e1.pack()
        e1.place(x=685, y=298)

        tk.Label(self, text="Enter gender:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=400)
        e2 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e2.pack()
        e2.place(x=685, y=398)

        tk.Label(self, text="Enter DOB:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=500)
        e3 = DateEntry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3", date_pattern="yyyy-mm-dd")
        e3.pack()
        e3.place(x=685, y=498)

        tk.Label(self, text="Enter phone:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=600)
        e4 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e4.pack()
        e4.place(x=685, y=598)

        tk.Label(self, text="Enter email:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=700)
        e5 = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e5.pack()
        e5.place(x=685, y=698)

        def add_customer():
            customer_name = e1.get()
            customer_gender = e2.get()
            customer_dob = e3.get()
            customer_phone = e4.get()
            customer_email = e5.get()

            if not customer_name or not customer_gender or not customer_dob or not customer_phone or not customer_email:
                tk.Label(self, text="All fields are required!", fg="red", bg="#69b5b5", font=("Product Sans", 30)).place(x=550, y=750)
                return

            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO customers (customer_name, customer_gender, customer_dob, customer_phone, customer_email)
                    VALUES (?, ?, ?, ?, ?)
                """, (customer_name, customer_gender, customer_dob, customer_phone, customer_email))
                conn.commit()

                cursor.execute("SELECT MAX(customer_id) FROM customers")
                new_customer_id = cursor.fetchone()[0]

                tk.Label(self, text=f"Customer added successfully!", fg="green", bg="#69b5b5", font=("Product Sans", 30)).place(x=550, y=900)

                conn.close()
            except Exception as e:
                print(f"Error: {e}")
                tk.Label(self, text=f"Error: {e}", fg="red", bg="#69b5b5", font=("Product Sans", 30)).place(x=550, y=750)


        tk.Button(self, text="Collect datasets", command=training_and_collection, relief=RAISED, width=15, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=45,y=800)
        tk.Button(self, text="Add customer", command=add_customer, relief=RAISED, width=15, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=45,y=900)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageEleven), height=86, width=90).place_configure(x=30, y=50)


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop() 