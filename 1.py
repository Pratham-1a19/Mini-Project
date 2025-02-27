# import smtplib
# import os
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# def send_email(to_email, subject, message):
#     # Email configuration
#     from_email = os.getenv("EMAIL_USER")
#     from_password = os.getenv("EMAIL_PASS")
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 587

#     # Create the email
#     msg = MIMEMultipart()
#     msg['From'] = from_email
#     msg['To'] = to_email
#     msg['Subject'] = subject

#     # Attach the message
#     msg.attach(MIMEText(message, 'plain'))

#     try:
#         # Connect to the SMTP server
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(from_email, from_password)

#         # Send the email
#         server.sendmail(from_email, to_email, msg.as_string())
#         print(f"Email sent to {to_email}")

#         # Disconnect from the server
#         server.quit()
#     except Exception as e:
#         print(f"Failed to send email: {e}")

# if __name__ == "__main__":
#     to_email = "devanshkansagra@gmail.com"
#     subject = "Test Email"
#     message = "This is a test email sent from Python."

#     send_email(to_email, subject, message)

import tkinter as tk

def show_selection():
    selected_texts = [text for text, var in checkboxes.items() if var.get()]
    print("Selected checkboxes:", selected_texts)

root = tk.Tk()
root.title("Checkbox Example")
root.geometry("300x200")

# Dictionary to store the text and corresponding BooleanVar
checkboxes = {}

# Create and place the checkboxes
texts = ["Option 1", "Option 2", "Option 3"]
for text in texts:
    var = tk.BooleanVar()
    checkbox = tk.Checkbutton(root, text=text, variable=var)
    checkbox.pack(pady=5)
    checkboxes[text] = var

# Create and place a button to print the selected checkbox texts
button = tk.Button(root, text="Submit", command=show_selection)
button.pack(pady=20)

root.mainloop()