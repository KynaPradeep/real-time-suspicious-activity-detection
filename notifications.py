from twilio.rest import Client
import os

# Load Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
ALERT_TO_NUMBER = os.getenv("ALERT_TO_NUMBER")

# Create Twilio client only if creds exist
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_TO_NUMBER]):
    raise RuntimeError("Twilio environment variables are not set correctly")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_sms_alert(event_type: str, label: str, confidence: float):
    """
    Sends an SMS alert for a critical event.
    """

    message = (
        f"SUSPICIOUS ACTIVITY DETECTED \n\n"
        f"Type: {event_type.upper()}\n"
        f"Label: {label.upper()}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Action recommended."
    )

    try:
        client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER
        )
        print("SMS sent successfully")

    except Exception as e:
        print(" SMS failed:", e)
