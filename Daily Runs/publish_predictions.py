#! python3
# publish_predictions_2018.py - Publishes predictions for the day by sending an email to stakeholders
# and posting on the BRC website
# RTS - 11/1/2017, UPDATE: 5/11/2018

# NOTE: Email and website credentials in the below functions have been redacted - RTS 01/17/2019

import pandas as pd
import numpy as np
import os
import re
import sys
import datetime
import smtplib
from string import Template
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from selenium import webdriver


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for fil in self.files:
            fil.write(obj)

    def flush(self):
        pass


def send_results_stakeholders(table, date, eml_file):
    def get_contacts(filename):
        df_contacts = pd.read_csv(filename)
        names = list(df_contacts['name'])
        emails = list(df_contacts['email'])

        return names, emails

    df_test = table
    # tab_style = '<style> .dataframe tbody tr:nth-child(even) { background-color: lightblue; } </style> '
    df_html = df_test.to_html(justify='left')  # , classes='df')
    df_html = re.sub('<th></th>', '<th>Beach</th>', df_html)
    df_html = df_html.replace('<tr>', '<tr style="text-align: left;">')  # Left justify all cells

    date_str = date

    login_usr = 'XXXX'  # credentials [DO NOT POST TO GITHUB]
    login_pwd = 'XXXX'

    contact_file = base_folder + '\email\email_contacts.csv'
    # contact_file = base_folder + '\email\email_contacts_test.csv'  # For testing
    template_file_html = base_folder + '\email\\results_email_template_html.txt'
    image_file = base_folder + '\email\HTB_logo.png'
    names, emails = get_contacts(contact_file)

    # Initiate Email #
    msg = MIMEMultipart('mixed')  # create a message
    # setup the parameters of the message
    msg['From'] = login_usr
    msg['To'] = login_usr
    msg['CC'] = 'calimodel@lists.stanford.edu'
    msg['BCC'] = ', '.join(emails)
    msg['Subject'] = 'NowCast Predictions - ' + date_str

    # Attach Text
    # Read HTML template file
    temp_file = open(template_file_html)  # HTML Message
    temp = Template(temp_file.read())
    temp_file.close()
    html_message = temp.substitute(DATE=date_str, BCC=', '.join(names), RESULTS=df_html)

    # Attach Image
    img = open(image_file, 'rb')  # HTB logo
    img_att = MIMEImage(img.read(), _subtype='png')
    img.close()
    img_att.add_header('Content-ID', '<image1>')

    # Attach CSV File
    table_file = os.path.join(email_table_folder, table_ext)
    fp = open(table_file, 'rb')
    csv_file = MIMEBase('application', 'csv')
    csv_file.set_payload(fp.read())
    fp.close()
    encoders.encode_base64(csv_file)
    ext_name = 'NowCast_Predictions_' + date_str.replace('/', '') + '.csv'
    csv_file.add_header('Content-Disposition', 'attachment;filename=%s' % ext_name)

    # Add in the message body and image
    msg.attach(MIMEText(html_message, 'html'))
    msg.attach(img_att)
    msg.attach(csv_file)

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


def upload_brc(upload_csv):
    upload_file = upload_csv

    usr = 'XXXX'  # credentials [DO NOT POST TO GITHUB]
    pwd = 'XXXX'

    url = 'admin.beachreportcard.org/admin/upload_nowcast'

    # Open Chrome #
    driver = webdriver.Chrome()
    driver.get('https://' + usr + ':' + pwd + '@' + url)  # Open admin site

    # Upload File #
    upload_elem = driver.find_element_by_xpath("//input[@type='file']")  # Find file button
    upload_elem.send_keys(upload_file)  # upload file
    submit_elem = driver.find_element_by_xpath("//button[@type='submit']")  # Find submit button
    submit_elem.click()

    # Identify bad upload (TODO Make more general)
    if 'Wrong File Format' in driver.page_source:
        print('\nERROR: File could not be uploaded to BRC. Please upload the file manually')
        error_list.append('BRC Upload Error: File could not be uploaded to BRC. Please upload the file manually')
    else:
        print('\nFile uploaded to BRC successfully')
    driver.close()


# SETUP
debug = 1  # 0 - production runs; 1 - debug mode (stop logging)
dont_send_email = 0
dont_post_brc = 0

date = datetime.date.today()  # - datetime.timedelta(days=1)
date_str = datetime.date.strftime(date, '%m/%d/%Y')

base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019'
import_folder = os.path.join(base_folder, 'import')
email_base_folder = os.path.join(base_folder, 'email')

error_list = []  # Initiate error list

if debug == 0:  # Log output if running (won't let you debug if logging)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = open(base_folder + '\\logs\\publish_log_' + date_str.replace('/', '') + '.log', 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

np.seterr(divide='ignore')  # Stop Divide vy zero warnings
print('- - - | Publishing NowCast Predictions for ' + date_str + ' | - - -')

# Email predictions to stakeholders
if dont_send_email == 0:
    email_table_folder = os.path.join(email_base_folder, 'tables')
    table_ext = 'NowCast_Email_Table' + date_str.replace('/', '') + '.csv'
    df_email = pd.read_csv(os.path.join(email_table_folder, table_ext), index_col=0)  # grab email table
    df_email.replace(np.nan, '', inplace=True)  # remove Nan in comments

    sent_email_folder = email_base_folder + '\sent_emails'
    email_ext = 'NowCast_Results_Email_' + date_str.replace('/', '') + '.eml'
    email_save_file = os.path.join(sent_email_folder, email_ext)
    if email_ext not in os.listdir(sent_email_folder):
        try:
            send_results_stakeholders(df_email, date_str, email_save_file)  # Email results for given day
        except:
            e = sys.exc_info()
            print('\nEmail Send Error: ' + str(e))
            error_list.append('Email Send Error: ' + str(e))

    else:
        print('\nResults email for ' + date_str + ' already sent.')

# Import to BRC
if dont_post_brc == 0:
    import_file = os.path.join(import_folder, 'NowCast_Import_' + date_str.replace('/', '') + '.csv')
    try:  # Upload import file to BRC site
        upload_brc(import_file)
    except:
        e = sys.exc_info()
        print('\nBRC Upload Error: ' + str(e))
        error_list.append('BRC Upload Error: ' + str(e))

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
