import smtplib

server = smtplib.SMTP('smtp..com', 587)

server.starttls()
server.login('22it439@bvmengineering.ac.in', 'demo@1234')
server.sendmail('22it439@bvmengineering.ac.in', 'pradpat1918@gmail.com', 'Hello, this is a test email.')
print('Email sent successfully')