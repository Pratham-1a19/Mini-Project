import smtplib

try:
    # Connect to Gmail's SMTP server
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login('22it439@bvmengineering.ac.in', 'your_app_password')  # Use App Password here
    server.sendmail('22it439@bvmengineering.ac.in', 'pradpat1918@gmail.com', 'Mail sent from Python')
    print('Mail sent successfully!')
except smtplib.SMTPAuthenticationError as e:
    print(f"Authentication failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    server.quit()