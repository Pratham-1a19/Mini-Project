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
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    

DB_PATH = "user.db"
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
        self._frame.configure(bg="#FFFCEA")
        self._frame.pack()

class StartPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#FFF1D5")

        def selection(value):
            if value == 1:
                master.switch_frame(PageSeven)
            elif value == 2:
                master.switch_frame(PageTen)
            elif value == 3:
                master.switch_frame(PageFifteen)
        
        load = Image.open("store.png")
        load = load.resize((650, 550), Image.Resampling.LANCZOS)
        render = ImageTk.PhotoImage(load)

        img = tk.Label(self, image=render)
        img.image = render
        img.place(x=525, y=170)

        radio = IntVar()
        tk.Label(self, text=f"The Super Market",  fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 55, "bold"), width=60).pack(pady=50)
        

        canvas = tk.Canvas(self, width=1727, height=400, bg="#FFFCEA", highlightthickness=0)
        canvas.place(x=0, y=750) 

        def create_custom_button(canvas, x, y, text, command):
            rect = canvas.create_rectangle(x, y, x + 400, y + 100, fill="#D9A760", outline="#B58A4A", width=3)
            inner_rect = canvas.create_rectangle(x + 5, y + 5, x + 395, y + 95, fill="#F3B65D", outline="#F3B65D")
            text_item = canvas.create_text(x + 200, y + 50, text=text, font=("Product Sans", 45, "bold"), fill="#e8edf3")

            canvas.tag_bind(rect, "<Button-1>", lambda event: command())
            canvas.tag_bind(inner_rect, "<Button-1>", lambda event: command())
            canvas.tag_bind(text_item, "<Button-1>", lambda event: command())


        create_custom_button(canvas, 150, 0, "Admin", lambda: selection(1))
        create_custom_button(canvas, 655, 0, "Staff", lambda: selection(2))
        create_custom_button(canvas, 1150, 0, "Customer", lambda: selection(3))

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
        tk.Label(self, text="General Store",  fg="#ffffff", bg="#F3B65D", width=60).pack(pady=50)
        tk.Radiobutton(self, text="New Customer", bg="#F3B65D", fg="#e8edf3", variable=radio, value=1, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=150, y=650)
        tk.Radiobutton(self, text="Existing Customer", bg="#F3B65D", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 50), highlightbackground="#000000", highlightcolor="#000000").place(x=950, y=650)
     

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
        tk.Label(self, text="Click below button for dataset collection",  fg="#f8f8f2", bg="#F3B65D", width=60).pack(pady=50)
        tk.Button(self, text="DATASET COLLECTION", command=training_and_collection, relief=RAISED, width=20, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=1100, y=500, anchor="e", height=50, width=500)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageFive), height=80, width=90).place_configure(x=30, y=50)
        
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
            names = ['None', 'Nishchay', 'Shiv']  # The name is to be added behind this
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
        tk.Label(self, text="Recognize Customer",  fg="#f8f8f2", bg="#F3B65D", width=60).pack(pady=50)
        tk.Button(self, text="RECOGNIZE USER", command=recognize_user, relief=RAISED, width=20, bg="#efe9d5", fg="#000000", font=("Product Sans", 30)).place_configure(x=1100,y=500,anchor="e",height=50,width=500)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageTen), height=80, width=90).place_configure(x=30, y=50)

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
        tk.Label(self, text=f"Welcome {id_text}",  fg="#f8f8f2", bg="#F3B65D", width=60).pack(pady=50)
        tk.Button(self,image=render,text="BACK",command=lambda: master.switch_frame(PageTwo),height=76,width=90).place_configure(x=30,y=50)
        
class PageFive(tk.Frame):
        def __init__(self, master):    
            super().__init__(master)
            self.pack(fill="both", expand=True)
            master.geometry('1727x1050+0+0')
            master.configure(bg="#69b5b5")

            bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
            normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")
            tk.Label(self, text=f"Login Page",  fg="#f8f8f2", bg="#F3B65D", width=60).pack(pady=50)
            
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
        tk.Label(self, text="Billing Page",  fg="#f8f8f2", bg="#F3B65D", width=60).pack(pady=50)
        
        self.checkboxes = {}
        self.entries = {}
        y_axis = 300

        texts = ["Milk", "Curd", "Buttermilk"]
        for text in texts:
            var = tk.BooleanVar()
            tk.Checkbutton(self, text=text, bg="#F3B65D", fg="#e8edf3", variable=var, width=10, height=1, bd=5, relief='raised', font=normal_font, highlightbackground="#000000", highlightcolor="#000000").place(x=200, y=y_axis)
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
        tk.Label(self, text=f"Admin Login Page",  fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        
        def callback4(id, pwd):
            if id != "admin":
                error_label.config(text="*admin id is incorrect", fg="red", bg="#FFFCEA")
                
            elif pwd != "admin123":
                error_label.config(text="*password is incorrect", fg="red", bg="#FFFCEA")

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

        error_label = tk.Label(self, text="", fg="red", bg="#FFFCEA", font=err_font)
        error_label.place(x=45, y=600)

        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(StartPage), height=80, width=90).place_configure(x=30, y=50)
        
class PageEight(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        self.render = ImageTk.PhotoImage(load)  # Store the image as an attribute

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Welcome to the Admin Page", fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        tk.Button(self, image=self.render, text="BACK", command=lambda: master.switch_frame(PageSeven), height=80, width=90).place_configure(x=30, y=50)

        def selection():
            a = radio.get()
            if a == 1:
                self.show_add_staff()
            elif a == 2:
                self.show_staff_list()
            elif a == 3:
                self.show_customer_list()

        radio = IntVar()
        tk.Radiobutton(self, text="Add New Staff Member", bg="#F3B65D", fg="#e8edf3", variable=radio, value=1, command=selection, width=18, height=1, bd=5, relief='raised', font=('Product Sans', 35), highlightbackground="#000000", highlightcolor="#000000").place(x=25, y=175)
        tk.Radiobutton(self, text="List of Staff", bg="#F3B65D", fg="#e8edf3", variable=radio, value=2, command=selection, width=18, height=1, bd=5, relief='raised', font=('Product Sans', 35), highlightbackground="#000000", highlightcolor="#000000").place(x=585, y=175)   
        tk.Radiobutton(self, text="List of Customer", bg="#F3B65D", fg="#e8edf3", variable=radio, value=3, command=selection, width=18, height=1, bd=5, relief='raised', font=('Product Sans', 35), highlightbackground="#000000", highlightcolor="#000000").place(x=1145, y=175)

        self.add_staff_frame = tk.Frame(self, bg="#FFFCEA")
        self.staff_list_frame = tk.Frame(self, bg="#FFFCEA")
        self.customer_list_frame = tk.Frame(self, bg="#FFFCEA")

        self.add_staff_frame.place_forget()
        self.staff_list_frame.place_forget()
        self.customer_list_frame.place_forget()

    def show_add_staff(self):
        self.customer_list_frame.place_forget() 
        self.add_staff_frame.place(x=0, y=300, relwidth=1, relheight=0.7)

        for widget in self.add_staff_frame.winfo_children():
            widget.destroy()

        normal_font = tkFont.Font(family="Product Sans", size=35, weight="normal")

        tk.Label(self.add_staff_frame, text="Staff Name:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place(x=200, y=0)
        e1 = tk.Entry(self.add_staff_frame, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e1.place(x=835, y=0)

        tk.Label(self.add_staff_frame, text="Gender:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place(x=200, y=100)
        e2 = tk.Entry(self.add_staff_frame, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e2.place(x=835, y=100)

        tk.Label(self.add_staff_frame, text="Date of birth:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place(x=200, y=200)
        e3 = DateEntry(self.add_staff_frame, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3", date_pattern="yyyy-mm-dd")
        e3.place(x=835, y=200)

        tk.Label(self.add_staff_frame, text="Email:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place(x=200, y=300)
        e4 = tk.Entry(self.add_staff_frame, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e4.place(x=835, y=300)

        tk.Label(self.add_staff_frame, text="Phone number:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place(x=200, y=400)
        e5 = tk.Entry(self.add_staff_frame, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        e5.place(x=835, y=400)

        def add_staff_to_db():
            staff_name = e1.get()
            staff_gender = e2.get()
            staff_dob = e3.get()
            staff_email = e4.get()
            staff_phone = e5.get()

            if not staff_name or not staff_gender or not staff_dob or not staff_email or not staff_phone:
                tk.Label(self.add_staff_frame, text="All fields are required!", fg="red", bg="#69b5b5", font=normal_font).place(x=45, y=550)
                return

            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()

                cursor.execute("SELECT MAX(staff_id) FROM staff_members")
                result = cursor.fetchone()
                next_id = (int(result[0][5:]) if result[0] else 0) + 1

                staff_id = f"staff{next_id}"

                cursor.execute("""
                    INSERT INTO staff_members (staff_id, staff_name, staff_gender, staff_dob, staff_email, staff_phone)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (staff_id, staff_name, staff_gender, staff_dob, staff_email, staff_phone))
                conn.commit()
                conn.close()

                tk.Label(self.add_staff_frame, text="Staff added successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=45, y=600)
            except Exception as e:
                print(f"Error: {e}")
                tk.Label(self.add_staff_frame, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=45, y=650)

        tk.Button(self.add_staff_frame, text="Add Staff", command=add_staff_to_db, relief=RAISED, width=19, fg="#22264b", bg='#e6cf8b', font=normal_font).place(x=200, y=500)

    def show_customer_list(self):
        self.add_staff_frame.place_forget()
        self.customer_list_frame.place(x=0, y=300, relwidth=1, relheight=0.7)  # Show customer list frame

        for widget in self.customer_list_frame.winfo_children():
            widget.destroy()

        normal_font = tkFont.Font(family="Product Sans", size=20, weight="normal")

        try:
            conn = sqlite3.connect("Store.db")
            cursor = conn.cursor()
            cursor.execute("SELECT customer_id, customer_name, customer_email, customer_phone FROM customers")
            customers = cursor.fetchall()
            conn.close()

            headers = ["Customer ID", "Name", "Email", "Phone"]
            for col, header in enumerate(headers):
                padx_value = (100, 5) if col == 0 else (5, 5)
                tk.Label(self.customer_list_frame, text=header, fg="#ffffff", bg="#553621", font=normal_font, width=20, anchor="w").grid(row=0, column=col, padx=padx_value, pady=5, sticky="w")

            for row, customer in enumerate(customers, start=1):
                for col, value in enumerate(customer):
                    padx_value = (100, 5) if col == 0 else (5, 5)  # Add left padding of 100 for the first column
                    tk.Label(self.customer_list_frame, text=value, fg="#22264b", bg="#e8edf3", font=normal_font, width=20, anchor="w").grid(row=row, column=col, padx=padx_value, pady=5, sticky="w")
        except Exception as e:
            print(f"Error: {e}")
            tk.Label(self.customer_list_frame, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).pack()
    
    def show_staff_list(self):
        self.customer_list_frame.place_forget()
        self.add_staff_frame.place(x=0, y=300, relwidth=1, relheight=0.7)  # Show staff list frame

        for widget in self.add_staff_frame.winfo_children():
            widget.destroy()

        normal_font = tkFont.Font(family="Product Sans", size=20, weight="normal")

        try:
            conn = sqlite3.connect("Store.db")
            cursor = conn.cursor()
            cursor.execute("SELECT staff_id, staff_name, staff_email, staff_phone FROM staff_members")
            staff_members = cursor.fetchall()
            conn.close()

            headers = ["Staff ID", "Name", "Email", "Phone"]
            for col, header in enumerate(headers):
                padx_value = (100, 5) if col == 0 else (5, 5)  # Add left padding of 100 for the first column
                tk.Label(self.add_staff_frame, text=header, fg="#ffffff", bg="#553621", font=normal_font, width=20, anchor="w").grid(row=0, column=col, padx=padx_value, pady=5, sticky="w")

            for row, staff in enumerate(staff_members, start=1):
                for col, value in enumerate(staff):
                    padx_value = (100, 5) if col == 0 else (5, 5)  # Add left padding of 100 for the first column
                    tk.Label(self.add_staff_frame, text=value, fg="#22264b", bg="#e8edf3", font=normal_font, width=20, anchor="w").grid(row=row, column=col, padx=padx_value, pady=5, sticky="w")
        except Exception as e:
            print(f"Error: {e}")
            tk.Label(self.add_staff_frame, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).pack()
    

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
        tk.Label(self, text="Add Staff Member",  fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        
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

            # Validate input fields
            if not staff_name or not staff_gender or not staff_dob or not staff_email or not staff_phone:
                tk.Label(self, text="All fields are required!", fg="red", bg="#69b5b5", font=normal_font).place(x=45, y=850)
                return

            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()

                cursor.execute("SELECT MAX(staff_id) FROM staff_members")
                result = cursor.fetchone()
                next_id = (int(result[0][5:]) if result[0] else 0) + 1

                staff_id = f"staff{next_id}"

                cursor.execute("""
                    INSERT INTO staff_members (staff_id, staff_name, staff_gender, staff_dob, staff_email, staff_phone)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (staff_id, staff_name, staff_gender, staff_dob, staff_email, staff_phone))
                conn.commit()
                conn.close()

                tk.Label(self, text="Staff added successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=45, y=900)
            except Exception as e:
                print(f"Error: {e}")
                tk.Label(self, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=60, y=850)

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
        tk.Label(self, text=f"Staff Login Page",  fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        
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
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(StartPage), height=80, width=90).place_configure(x=30, y=50)

        error_label = tk.Label(self, text="", fg="red", bg="#FFFCEA", font=err_font)
        error_label.place(x=45, y=600)

class PageEleven(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        self.render = ImageTk.PhotoImage(load)

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Welcome to the Staff Page", fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        tk.Button(self, image=self.render, text="BACK", command=lambda: master.switch_frame(PageTen), height=80, width=90).place_configure(x=30, y=50)

        self.add_item_frame = tk.Frame(self, bg="#FFFCEA")
        self.new_customer_frame = tk.Frame(self, bg="#FFFCEA")
        self.existing_customer_frame = tk.Frame(self, bg="#FFFCEA")


        self.add_item_frame.place_forget()
        self.new_customer_frame.place_forget()
        self.existing_customer_frame.place_forget()


        def selection():
            a = radio.get()
            if a == 1:
                self.show_add_item()
        # Radio buttons at the top
        radio = IntVar()
        tk.Radiobutton(self, text="Add Items", bg="#F3B65D", fg="#e8edf3", variable=radio, value=1, command=selection, width=46, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=50, y=175)

    def show_add_item(self):
        """Show the Add Item content."""
        self.new_customer_frame.place_forget()
        self.existing_customer_frame.place_forget()
        self.add_item_frame.place(x=0, y=150, relwidth=1, relheight=0.8)

        # Clear the frame before adding new widgets
        for widget in self.add_item_frame.winfo_children():
            widget.destroy()

        normal_font = tkFont.Font(family="Product Sans", size=30, weight="normal")

        # Dropdown for item category
        categories = ["Please Select Category", "Baby Products", "Bakery Products", "Dairy Products", "Fresh Products", "Frozen Food", "Packaged Products", "Personal Care", "Pet Supplies"]
        category_var = StringVar(self.add_item_frame)
        category_var.set(categories[0])

        tk.Label(self.add_item_frame, text="Item Category : ", fg="#22264b", bg="#e8edf3", font=normal_font, width=11).place(x=50, y=150)
        category_menu = tk.OptionMenu(self.add_item_frame, category_var, *categories)
        category_menu.config(font=normal_font, bg="#e8edf3", fg="#22264b")
        category_menu.place(x=350, y=150)

        # Add item form
        tk.Label(self.add_item_frame, text="Item Name : ", fg="#22264b", bg="#e8edf3", font=normal_font, width=11).place(x=50, y=250)
        item_name_entry = tk.Entry(self.add_item_frame, font=normal_font)
        item_name_entry.place(x=350, y=250)

        tk.Label(self.add_item_frame, text="Item Price : ", fg="#22264b", bg="#e8edf3", font=normal_font, width=11).place(x=50, y=350)
        item_price_entry = tk.Entry(self.add_item_frame, font=normal_font)
        item_price_entry.place(x=350, y=350)

        tk.Label(self.add_item_frame, text="Item Quantity : ", fg="#22264b", bg="#e8edf3", font=normal_font, width=11).place(x=50, y=450)
        item_quantity_entry = tk.Entry(self.add_item_frame, font=normal_font)
        item_quantity_entry.place(x=350, y=450)

        def add_item_to_db():
            """Add item to the database and notify users."""
            category = category_var.get()
            item_name = item_name_entry.get()
            item_price = item_price_entry.get()
            item_quantity = item_quantity_entry.get()

            if category == "Please Select Category" or not item_name or not item_price or not item_quantity:
                tk.Label(self.add_item_frame, text="All fields are required!", fg="red", bg="#69b5b5", font=normal_font).place(x=50, y=650)
                return

            try:
                # Insert item into the database
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()
                cursor.execute(f"""
                    INSERT INTO "{category}" (product_name, product_qty, product_prize)
                    VALUES (?, ?, ?)
                """, (item_name, item_quantity, item_price))
                conn.commit()

                # Notify users with matching preferences
                cursor.execute("SELECT customer_email, customer_name, customer_preference FROM customers")
                customers = cursor.fetchall()

                matching_customers = []
                for email, name, preferences in customers:
                    try:
                        if preferences:
                            preference_list = json.loads(preferences)
                            if category in preference_list:
                                matching_customers.append((email, name))
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in preferences for customer {name}: {preferences}")

                for email, name in matching_customers:
                    send_email(email, name, category, item_name, item_price)

                conn.close()
                tk.Label(self.add_item_frame, text="Item added and emails sent successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=50, y=650)
            except Exception as e:
                print(f"Error: {e}")
                tk.Label(self.add_item_frame, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=200, y=500)

        def send_email(email, name, category, product_name, product_price):
            try:
                SMTP_SERVER = "smtp.gmail.com"
                sender_email = "22it439@bvmengineering.ac.in"
                sender_password = "dummy#1234"
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

        tk.Button(self.add_item_frame, text="Add Item", command=add_item_to_db, fg="#22264b", bg="#e6cf8b", font=normal_font, width=31).place(x=50, y=550)

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
        tk.Label(self, text="Add item to the store",  fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        
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
                SMTP_SERVER = "smtp.gmail.com"
                sender_email = "22it439@bvmengineering.ac.in"
                sender_password = "dummy#1234"
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
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageEleven), height=80, width=90).place_configure(x=30, y=50)

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

        tk.Label(self, text="New Customer Details",  fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)

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
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageEleven), height=80, width=90).place_configure(x=30, y=50)

class PageFourteen(tk.Frame):
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

        tk.Label(self, text="Billing Page", fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)

        tk.Label(self, text="Enter Customer Id:", fg="#22264b", bg="#e8edf3", width=20, height=1, font=normal_font, anchor="center").place_configure(x=45, y=200)
        customer_id_entry = tk.Entry(master, width=20, font=(normal_font), fg="#22264b", bg="#e8edf3")
        customer_id_entry.pack()
        customer_id_entry.place(x=685, y=200)

        # Frame for product list
        product_frame = tk.Frame(self, bg="#FFFCEA")
        product_frame.place(x=45, y=400, relwidth=0.9, relheight=0.5)

        # Scrollbar for the product frame
        canvas = tk.Canvas(product_frame, bg="#FFFCEA")
        scrollbar = tk.Scrollbar(product_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#FFFCEA")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dictionary to store quantity inputs and checkboxes
        self.quantity_inputs = {}
        self.checkboxes = {}

        # Table names
        table_names = [
            "Baby Products", "Bakery Products", "Dairy Products", "Fresh Products",
            "Frozen Food", "Packaged Products", "Personal Care", "Pet Supplies"
        ]

        # Fetch products from the database and display them
        try:
            conn = sqlite3.connect("Store.db")
            cursor = conn.cursor()

            row_index = 0
            for table_name in table_names:
                # Display category name
                tk.Label(scrollable_frame, text=table_name, fg="#22264b", bg="#e8edf3", font=normal_font).grid(row=row_index, column=0, columnspan=3, pady=10, sticky="w")
                row_index += 1

                # Fetch products in the category
                cursor.execute(f"SELECT product_name, product_prize FROM '{table_name}'")
                products = cursor.fetchall()

                for product_name, product_price in products:
                    # Checkbox for selecting the product
                    var = tk.BooleanVar()
                    tk.Checkbutton(scrollable_frame, text=f"{product_name} (Rs. {product_price})", variable=var, bg="#FFFCEA", fg="#22264b", font=normal_font).grid(row=row_index, column=0, padx=10, pady=5, sticky="w")
                    self.checkboxes[(table_name, product_name)] = var

                    # Input box for quantity
                    quantity_var = tk.StringVar()
                    tk.Entry(scrollable_frame, textvariable=quantity_var, width=5, font=normal_font).grid(row=row_index, column=1, padx=10, pady=5)
                    self.quantity_inputs[(table_name, product_name)] = quantity_var

                    row_index += 1

            conn.close()
        except Exception as e:
            print(f"Error fetching products: {e}")
            tk.Label(self, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=45, y=900)

        # Submit button
        tk.Button(self, text="Submit", command=lambda: self.submit_billing(customer_id_entry.get()), relief=RAISED, width=15, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place(x=45, y=900)
        tk.Button(self, image=render, text="BACK", command=lambda: master.switch_frame(PageFifteen), height=80, width=90).place_configure(x=30, y=50)

    def submit_billing(self, customer_id):
        """Handle billing submission."""
        try:
            conn = sqlite3.connect("Store.db")
            cursor = conn.cursor()

            total_amount = 0
            purchased_items = []
            purchased_categories = set()  # To track purchased categories

            for (table_name, product_name), checkbox_var in self.checkboxes.items():
                if checkbox_var.get():  # Check if the product is selected
                    quantity = self.quantity_inputs[(table_name, product_name)].get()
                    if quantity.isdigit() and int(quantity) > 0:
                        quantity = int(quantity)

                        cursor.execute(f"SELECT product_prize, product_qty FROM '{table_name}' WHERE product_name = ?", (product_name,))
                        product_data = cursor.fetchone()
                        if not product_data:
                            continue
                        product_price, current_qty = product_data
                        current_qty = int(current_qty)

                        if current_qty < quantity:
                            tk.Label(self, text=f"Insufficient stock for {product_name}!", fg="red", bg="#69b5b5", font=("Product Sans", 20)).place(x=45, y=850)
                            continue

                        # Calculate total for the product
                        total_price = product_price * quantity
                        total_amount += total_price

                        purchased_items.append((product_name, quantity, total_price))
                        purchased_categories.add(table_name)  # Add category to the set

                        # Update product quantity in the database
                        new_qty = current_qty - quantity
                        cursor.execute(f"UPDATE '{table_name}' SET product_qty = ? WHERE product_name = ?", (new_qty, product_name))

                        # Remove the product from the table if the quantity becomes 0
                        if new_qty == 0:
                            cursor.execute(f"DELETE FROM '{table_name}' WHERE product_name = ?", (product_name,))

            # Update customer preferences
            if purchased_categories:
                cursor.execute("SELECT customer_preference FROM customers WHERE customer_id = ?", (customer_id,))
                result = cursor.fetchone()
                if result and result[0]:
                    try:
                        current_preferences = set(json.loads(result[0]))
                    except json.JSONDecodeError:
                        current_preferences = set()
                else:
                    current_preferences = set()

                # Add purchased categories to the customer's preferences
                updated_preferences = list(current_preferences.union(purchased_categories))
                cursor.execute("UPDATE customers SET customer_preference = ? WHERE customer_id = ?", (json.dumps(updated_preferences), customer_id))

            conn.commit()
            conn.close()

            # Display billing summary
            if purchased_items:
                summary = f"Customer ID: {customer_id}\n\n"
                summary += "Purchased Items:\n"
                for product_name, quantity, total_price in purchased_items:
                    summary += f"{product_name} - {quantity} pcs - Rs. {total_price}\n"
                summary += f"\nTotal Amount: Rs. {total_amount}"

                tk.Label(self, text="Billing Successful!", fg="green", bg="#69b5b5", font=("Product Sans", 30)).place(x=700, y=900)
                print(summary)  # You can replace this with saving the summary to a file or database
            else:
                tk.Label(self, text="No items selected!", fg="red", bg="#69b5b5", font=("Product Sans", 30)).place(x=700, y=900)

        except Exception as e:
            print(f"Error during billing: {e}")
            tk.Label(self, text=f"Error: {e}", fg="red", bg="#69b5b5", font=("Product Sans", 30)).place(x=400, y=900)

class PageFifteen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.geometry('1727x1050+0+0')
        master.configure(bg="#69b5b5")

        load = Image.open("back-arrow.png")
        load = load.resize((80, 80), Image.Resampling.LANCZOS)
        self.render = ImageTk.PhotoImage(load)

        bold_font = tkFont.Font(family='Product Sans', size=55, weight='bold')
        tk.Label(self, text="Welcome to the Customer Page", fg="#f8f8f2", bg="#553621", font=("Moldie Demo", 50, "bold"), width=60).pack(pady=50)
        tk.Button(self, image=self.render, text="BACK", command=lambda: master.switch_frame(StartPage), height=80, width=90).place_configure(x=30, y=50)

        self.add_item_frame = tk.Frame(self, bg="#FFFCEA")
        self.new_customer_frame = tk.Frame(self, bg="#FFFCEA")
        self.existing_customer_frame = tk.Frame(self, bg="#FFFCEA")


        self.add_item_frame.place_forget()
        self.new_customer_frame.place_forget()
        self.existing_customer_frame.place_forget()


        def selection():
            a = radio.get()
            if a == 2:
                self.show_new_customer()
            elif a == 3:
                self.show_existing_customer()
            elif a == 4:
                master.switch_frame(PageFourteen)

        # Radio buttons at the top
        radio = IntVar()
        tk.Radiobutton(self, text="New Customer", bg="#F3B65D", fg="#e8edf3", variable=radio, value=2, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=60, y=175)
        tk.Radiobutton(self, text="Existing Customer", bg="#F3B65D", fg="#e8edf3", variable=radio, value=3, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=600, y=175)
        tk.Radiobutton(self, text="Billing", bg="#F3B65D", fg="#e8edf3", variable=radio, value=4, command=selection, width=15, height=1, bd=5, relief='raised', font=('Product Sans', 40), highlightbackground="#000000", highlightcolor="#000000").place(x=1140, y=175)

    def show_new_customer(self):
        """Show the New Customer content."""
        self.add_item_frame.place_forget()
        self.existing_customer_frame.place_forget()
        self.new_customer_frame.place(x=0, y=150, relwidth=1, relheight=0.8)

        # Clear the frame before adding new widgets
        for widget in self.new_customer_frame.winfo_children():
            widget.destroy()

        normal_font = tkFont.Font(family="Product Sans", size=30, weight="normal")

        # Labels and Entry fields for customer details
        tk.Label(self.new_customer_frame, text="Customer Name:", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=150)
        customer_name_entry = tk.Entry(self.new_customer_frame, font=normal_font)
        customer_name_entry.place(x=400, y=150)

        tk.Label(self.new_customer_frame, text="Customer Gender:", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=250)
        customer_gender_entry = tk.Entry(self.new_customer_frame, font=normal_font)
        customer_gender_entry.place(x=400, y=250)

        tk.Label(self.new_customer_frame, text="Customer DOB:", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=350)
        customer_dob_entry = DateEntry(self.new_customer_frame, font=normal_font, fg="#22264b", bg="#e8edf3", date_pattern="yyyy-mm-dd")
        customer_dob_entry.place(x=400, y=350)

        tk.Label(self.new_customer_frame, text="Customer Phone:", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=450)
        customer_phone_entry = tk.Entry(self.new_customer_frame, font=normal_font)
        customer_phone_entry.place(x=400, y=450)

        tk.Label(self.new_customer_frame, text="Customer Email:", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=550)
        customer_email_entry = tk.Entry(self.new_customer_frame, font=normal_font)
        customer_email_entry.place(x=400, y=550)

        def add_customer():
            """Add a new customer to the database."""
            customer_name = customer_name_entry.get()
            customer_gender = customer_gender_entry.get()
            customer_dob = customer_dob_entry.get()
            customer_phone = customer_phone_entry.get()
            customer_email = customer_email_entry.get()

            if not customer_name or not customer_gender or not customer_dob or not customer_phone or not customer_email:
                tk.Label(self.new_customer_frame, text="All fields are required!", fg="red", bg="#69b5b5", font=normal_font).place(x=50, y=650)
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

                tk.Label(self.new_customer_frame, text=f"Customer added successfully! ID: {new_customer_id}", fg="green", bg="#69b5b5", font=normal_font).place_configure(x=950, y=650)

                conn.close()
            except Exception as e:
                print(f"Error: {e}")
                tk.Label(self.new_customer_frame, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=50, y=750)

        def training_and_collection():
            """Collect datasets for the new customer."""
            try:
                conn = sqlite3.connect("Store.db")
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(customer_id) FROM customers")
                result = cursor.fetchone()
                if result and result[0]:
                    face_id = result[0]
                else:
                    tk.Label(self.new_customer_frame, text="No customer found in the database!", fg="red", bg="#69b5b5", font=normal_font).place_configure(x=950, y=800)
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

                tk.Label(self.new_customer_frame, text="Dataset trained successfully!", fg="green", bg="#69b5b5", font=normal_font).place(x=50, y=850)
            except Exception as e:
                print(f"Error: {e}")
                tk.Label(self.new_customer_frame, text=f"Error: {e}", fg="red", bg="#69b5b5", font=normal_font).place(x=50, y=900)

        # Buttons for adding customer and collecting datasets
        tk.Button(self.new_customer_frame, text="Add Customer", command=add_customer, relief=RAISED, width=15, fg="#22264b", bg='#44475a', font=normal_font).place(x=50, y=650)
        tk.Button(self.new_customer_frame, text="Collect Datasets", command=training_and_collection, relief=RAISED, width=15, fg="#22264b", bg='#44475a', font=normal_font).place(x=500, y=650)

    def show_existing_customer(self):
        self.add_item_frame.place_forget()
        self.new_customer_frame.place_forget()
        self.existing_customer_frame.place(x=0, y=150, relwidth=1, relheight=0.8)

        for widget in self.existing_customer_frame.winfo_children():
            widget.destroy()

        normal_font = tkFont.Font(family="Product Sans", size=30, weight="normal")

        self.detect_button = tk.Button(
        self.existing_customer_frame,
        text="Detect Existing Customer",
        command=lambda: self.run_detect_faces_from_dataset(),
        fg="#22264b",
        bg="#FFF1D5",
        font=normal_font
        )
        self.detect_button.place_configure(x=50, y=200)

        tk.Label(self.existing_customer_frame, text="OR", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=300)
        tk.Label(self.existing_customer_frame, text="Enter customer id : ", fg="#22264b", bg="#e8edf3", font=normal_font).place(x=50, y=400)
        customer_name_entry = tk.Entry(self.existing_customer_frame, font=normal_font)
        customer_name_entry.place(x=400, y=400)
        tk.Button(self.existing_customer_frame, text="Detect", relief=RAISED, width=7, fg="#22264b", bg='#44475a', font=("Product Sans", 30)).place_configure(x=940,y=400)
        
    def detect_faces_from_dataset(self, dataset_path, cascade_path, db_path):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(cascade_path)

        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            face_samples = []
            ids = []
            id_to_name = {}

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT customer_id, customer_name FROM customers")
            id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()

            for image_path in image_paths:
                try:
                    # Convert image to grayscale
                    PIL_img = Image.open(image_path).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')

                    # Extract ID from the image filename
                    id = int(os.path.split(image_path)[-1].split(".")[1])

                    # Detect faces in the image
                    faces = detector.detectMultiScale(img_numpy)
                    for (x, y, w, h) in faces:
                        face_samples.append(img_numpy[y:y+h, x:x+w])
                        ids.append(id)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

            return face_samples, ids, id_to_name

        faces, ids, id_to_name = get_images_and_labels(dataset_path)
        return faces, ids, id_to_name
    
    def run_detect_faces_from_dataset(self):
        cascade_path = "haarcascade_frontalface_default.xml"
        db_path = "Store.db"

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("trainer.yml")  # Load trained model
            face_cascade = cv2.CascadeClassifier(cascade_path)

            cam = cv2.VideoCapture(0)
            detected_customer_id = None

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                    if confidence < 60:  # Confidence threshold
                        detected_customer_id = id_
                    else:
                        detected_customer_id = None

                    break  # Stop after first face

                cv2.imshow('Recognizing Customer', img)

                if cv2.waitKey(1) == ord('q') or detected_customer_id is not None:
                    break

            cam.release()
            cv2.destroyAllWindows()

            if detected_customer_id:
                # Fetch name from ID
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT customer_name FROM customers WHERE customer_id = ?", (detected_customer_id,))
                row = cursor.fetchone()
                conn.close()

                detected_customer_name = row[0] if row else "Unknown"

                self.display_customer_info(detected_customer_id, detected_customer_name)

            else:
                tk.Label(
                    self.existing_customer_frame,
                    text="No customer detected. Please try again.",
                    fg="red",
                    bg="#69b5b5",
                    font=("Product Sans", 20)
                ).place(x=50, y=300)

        except Exception as e:
            print(f"Error: {e}")
            tk.Label(
                self.existing_customer_frame,
                text=f"Error: {e}",
                fg="red",
                bg="#69b5b5",
                font=("Product Sans", 20)
            ).place(x=50, y=300)
    
    def display_customer_info(self, customer_id, customer_name):
        db_path = "Store.db"

        # Clear existing widgets
        for widget in self.existing_customer_frame.winfo_children():
            if isinstance(widget, (tk.Entry, tk.Button, tk.Label)):
                widget.destroy()

        # Welcome message
        tk.Label(
            self.existing_customer_frame,
            text=f"Welcome, {customer_name}! Your ID is: {customer_id}",
            fg="green",
            bg="#FFFCEA",
            font=("Product Sans", 30)
        ).place(x=50, y=150)

        # Fetch preferences from database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT customer_preference FROM customers WHERE customer_id = ?", (customer_id,))
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                preferences = json.loads(result[0])  # Expecting a JSON list like ["Shirts", "Watches"]
                row_index = 250

                for category in preferences:
                    # Display category label
                    tk.Label(
                        self.existing_customer_frame,
                        text=f"Category: {category}",
                        fg="#22264b",
                        bg="#e8edf3",
                        font=("Product Sans", 25)
                    ).place(x=50, y=row_index)
                    row_index += 50

                    # Fetch products from category table
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    try:
                        cursor.execute(f"SELECT product_name, product_prize FROM '{category}' LIMIT 2")
                        products = cursor.fetchall()
                    except sqlite3.OperationalError:
                        products = []
                    conn.close()

                    if products:
                        for product_name, product_price in products:
                            tk.Label(
                                self.existing_customer_frame,
                                text=f"{product_name} - Rs. {product_price}",
                                fg="#22264b",
                                bg="#FFFCEA",
                                font=("Product Sans", 20)
                            ).place(x=100, y=row_index)
                            row_index += 50
                    else:
                        tk.Label(
                            self.existing_customer_frame,
                            text="No products available in this category.",
                            fg="red",
                            bg="#FFFCEA",
                            font=("Product Sans", 20)
                        ).place(x=100, y=row_index)
                        row_index += 30
            else:
                tk.Label(
                    self.existing_customer_frame,
                    text="No preferences found for this customer.",
                    fg="red",
                    bg="#69b5b5",
                    font=("Product Sans", 20)
                ).place(x=50, y=400)

        except Exception as e:
            tk.Label(
                self.existing_customer_frame,
                text=f"Error loading preferences: {e}",
                fg="red",
                bg="#69b5b5",
                font=("Product Sans", 20)
            ).place(x=50, y=400)




if __name__ == "__main__":
    app = SampleApp()
    app.mainloop() 