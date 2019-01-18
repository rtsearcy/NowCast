#! python3
# send_np_predictions_2018.py - Sends generic no predictions email to stakeholders if the NowCast system is offline
# RTS - 11/1/2017, UPDATE: 5/11/2018

# NOTE: Email credentials in the below function have been redacted - RTS 01/17/2019

import pandas as pd
import numpy as np
import os
import sys
import datetime
import smtplib
from string import Template
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for fil in self.files:
            fil.write(obj)

    def flush(self):
        pass


def send_no_predictions(date, eml_file):
    def get_contacts(filename):
        df_contacts = pd.read_csv(filename)
        names = list(df_contacts['name'])
        emails = list(df_contacts['email'])

        return names, emails

    date_str = date

    # Credentials [DON'T POST ON GITHUB]
    login_usr = 'XXX'
    login_pwd = 'XXX'

    contact_file = base_folder + '\email\email_contacts.txt'
    # contact_file = base_folder + '\email\email_contacts_test.csv'  # for testing

    template_file_html = base_folder + '\email\\no_predictions_email_template_html.txt'
    image_file = base_folder + '\email\HTB_logo.png'
    names, emails = get_contacts(contact_file)

    # Initiate Email #
    msg = MIMEMultipart()  # create a message
    # setup the parameters of the message
    msg['From'] = login_usr
    msg['To'] = login_usr
    msg['CC'] = 'calimodel@lists.stanford.edu'
    msg['BCC'] = ', '.join(emails)
    msg['Subject'] = 'NowCast Predictions for ' + date_str + ' Unavailable'

    # Text
    # Read HTML template file
    temp_file = open(template_file_html)  # HTML Message
    temp = Template(temp_file.read())
    temp_file.close()
    html_message = temp.substitute(DATE=date_str, BCC=', '.join(names))
    # html_message = html_message.replace('<tr>', '<tr style="text-align: left;">')  # Left Justify All Cells

    # Image
    img = open(image_file, 'rb')  # HTB logo
    img_att = MIMEImage(img.read(), _subtype='png')
    img.close()
    img_att.add_header('Content-ID', '<image1>')

    # Add in the message body and image
    msg.attach(MIMEText(html_message, 'html'))
    msg.attach(img_att)

    # Send the message
    s = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
    s.starttls()
    s.login(login_usr, login_pwd)
    s.send_message(msg)
    s.quit()

    # Save message
    save_file = eml_file
    gen = email.generator.Generator(open(save_file, 'w'))
    gen.flatten(msg)

    return print('\nEmail with results for ' + date_str + ' sent.')


# SETUP
debug = 0  # 0 - production runs; 1 - debug mode (stop logging)

date = datetime.date.today()  # - datetime.timedelta(days=1)
date_str = datetime.date.strftime(date, '%m/%d/%Y')

base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019'
email_base_folder = os.path.join(base_folder, 'email')

error_list = []  # Initiate error list

if debug == 0:  # Log output if running (won't let you debug if logging)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = open(base_folder + '\\logs\\no_pred_log_' + date_str.replace('/', '') + '.log', 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

np.seterr(divide='ignore')  # Stop Divide vy zero warnings
print('- - - | Sending No Predictions Email for ' + date_str + ' | - - -')

# Email to stakeholders
sent_email_folder = email_base_folder + '\sent_emails'
email_ext = 'NowCast_Results_Email_' + date_str.replace('/', '') + '.eml'
email_save_file = os.path.join(sent_email_folder, email_ext)

# Prevent Resend
if email_ext not in os.listdir(sent_email_folder):
    try:
        send_no_predictions(date_str, email_save_file)  # Email results for given day
    except:
        e = sys.exc_info()
        print('\nEmail Send Error: ' + str(e))
        error_list.append('Email Send Error: ' + str(e))

else:
    print('\nResults email for ' + date_str + ' already sent.')

# Errors
if len(error_list) > 0:
    print('\nErrors: \n')
    for e in error_list:
        print(e)
else:
    print('\n- - - No Errors - - -')

if debug == 0:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()

# TODO if errors, email modeler there was an error in sending/uploading
