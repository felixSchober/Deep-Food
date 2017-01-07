import data_io.settings as Settings
import cgi
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from email.header         import Header
import os
import smtplib
from email.MIMEBase import MIMEBase
from email import Encoders

def are_mails_enabled():
    return not (Settings.G_MAIL_FROM == "" or Settings.G_MAIL_TO == "" or Settings.G_MAIL_PASSWD == "" or Settings.G_MAIL_SERVER == "" or Settings.G_MAIL_USER == "")

def send_mail(content, subject, embeddedImagesPaths=[], attachedFilePaths=[]):
    if are_mails_enabled():
        ms = MailService()
        ms.send_email(content, subject, embeddedImagesPaths, attachedFilePaths)

class MailService(object):
    """Class to send mails."""

    def __init__(self):
        pass
    
    def __attach_image(self, img_dict):
        with open(img_dict['path'], 'rb') as file:
            msg_image = MIMEImage(file.read(), name = os.path.basename(img_dict['path']))
        msg_image.add_header('Content-ID', '<{}>'.format(img_dict['cid']))
        return msg_image

    def __attach_file(self, filename):
        part = MIMEBase('application', 'octect-stream')
        part.set_payload(open(filename, 'rb').read())
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename=%s' % os.path.basename(filename))
        return part

    def __generate_email(self, content, subject="", embeddedImages=[], attachedFilePaths=[]):
        msg =MIMEMultipart('related')
        msg['Subject'] = Header(u'[TUM BA] Report ' + subject, 'utf-8')
        msg['From'] = Settings.G_MAIL_FROM
        msg['To'] = Settings.G_MAIL_TO
        msg_alternative = MIMEMultipart('alternative')
        msg_text = MIMEText(u'enable HTML to view full message.', 'plain', 'utf-8')
        msg_alternative.attach(msg_text)
        msg.attach(msg_alternative)
        msg_html = u'[BA @ Felix Schober]<br><br><h1>Content</h1>' + content
        
        # embedd Images into HTML
        if embeddedImages:
            msg_html += u'<h1>Images</h1>'
            for img in embeddedImages:
                msg_html += u'<div dir="ltr">''<img src="cid:{cid}" alt="{alt}"><br></div>'.format(alt=cgi.escape(img['title'], quote=True), **img)

        msg_html = MIMEText(msg_html, 'html', 'utf-8')
        msg_alternative.attach(msg_html)

        if embeddedImages:
            for img in embeddedImages:
                msg.attach(self.__attach_image(img))
    
        if attachedFilePaths:
            for filePath in attachedFilePaths:
                msg.attach(self.__attach_file(filePath))

        return msg

    def send_email(self, content, subject, embeddedImagesPaths=[], attachedFilePaths=[]):     
        if not are_mails_enabled():
            raise Exception("Mails are not setup. Go to settings and setup a mail account.") 
        imgs = []
        for path in embeddedImagesPaths:
            imgs.append(dict(title = 'Image', path = path, cid = str(uuid.uuid4())))
        msg = self.__generate_email(content, subject, imgs, attachedFilePaths)

        smtp = smtplib.SMTP()
        smtp.connect(Settings.G_MAIL_SERVER)
        try:
            smtp.login(Settings.G_MAIL_USER, Settings.G_MAIL_PASSWD)
        except Exception, e:
            raise Exception("Could not authenticate.",e)
        smtp.sendmail(Settings.G_MAIL_FROM, [Settings.G_MAIL_TO], msg.as_string())
        smtp.quit()