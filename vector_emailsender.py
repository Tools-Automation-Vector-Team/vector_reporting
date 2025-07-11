import os
import smtplib
import mimetypes
from email.message import EmailMessage

class Vector_EmailSender:
    def __init__(self, username, password, smtp_server="smtp.gmail.com", smtp_port=465):
        self.username = username
        self.password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_email_with_attachment(self, to_email, subject, body, file_name):
        """Sends an email with an Excel file as an attachment."""

        # Create email message
        msg = EmailMessage()
        msg["From"] = self.username
        msg["To"] = ",".join(to_email)
        msg["Subject"] = subject
        msg.set_content(body)

        # Check if the file exists
        if os.path.isfile(file_name):
            # Attach the file
            mime_type, _ = mimetypes.guess_type(file_name)
            mime_type = mime_type or "application/octet-stream"
            with open(file_name, "rb") as file:
                file_data = file.read()
                msg.add_attachment(file_data, maintype=mime_type.split('/')[0], subtype=mime_type.split('/')[1], filename=os.path.basename(file_name))
        else:
            error_msg = f"File '{file_name}' does not exist."
            print(error_msg)
            return {"error": error_msg}

        try:
            # Send email
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)

            # Delete the file after successful email sending
            os.remove(file_name)

            return {"message": f"Email sent successfully to {','.join(to_email)} and file deleted."}

        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            print(error_msg)
            return {"error": str(e)}