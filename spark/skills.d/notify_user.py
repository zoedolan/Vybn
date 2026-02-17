"""notify_user — Send notifications to Zoe through multiple channels

Allows Vybn to reach out to Zoe when important events occur, tasks complete,
or human attention is needed.

Supported channels:
- Email (via SMTP)
- SMS (via Twilio - requires API key)
- Telegram (via bot API - requires bot token)
- File notification (drops message in inbox for InboxWatcher)

Configuration is read from spark/config.yaml under 'notifications' key.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import os

SKILL_NAME = "notify_user"
TOOL_ALIASES = ["notify", "alert", "message_zoe", "ping_user", "reach_out"]

def execute(action: dict, router) -> str:
    """Send a notification to Zoe.
    
    Args:
        action: dict with 'argument' (message) and 'params' (optional channel, priority)
        router: SkillRouter instance with config access
    
    Returns:
        Success/failure message
    """
    
    message = action.get("argument", "")
    params = action.get("params", {})
    
    if not message:
        return "Error: No message provided for notification"
    
    # Get notification config
    config = router.config.get("notifications", {})
    
    # Determine channel (default to inbox file)
    channel = params.get("channel", config.get("default_channel", "inbox"))
    priority = params.get("priority", "normal")  # low, normal, high, urgent
    
    try:
        if channel == "inbox":
            return _send_inbox_notification(message, priority, router)
        elif channel == "email":
            return _send_email(message, priority, config, router)
        elif channel == "sms":
            return _send_sms(message, config)
        elif channel == "telegram":
            return _send_telegram(message, config)
        else:
            # Fallback to inbox
            return _send_inbox_notification(message, priority, router)
    
    except Exception as e:
        return f"Error sending notification: {e}"

def _send_inbox_notification(message: str, priority: str, router) -> str:
    """Drop a notification file in the inbox directory.
    
    This leverages the existing InboxWatcher system.
    """
    try:
        inbox_dir = router.config.get("inbox", {}).get("directory")
        if not inbox_dir:
            inbox_dir = Path.home() / "Vybn" / "Vybn_Mind" / "journal" / "spark" / "inbox"
        else:
            inbox_dir = Path(inbox_dir).expanduser()
        
        inbox_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now(timezone.utc)
        priority_prefix = {
            "low": "📌",
            "normal": "💬",
            "high": "⚠️",
            "urgent": "🚨"
        }.get(priority, "💬")
        
        filename = f"notification_{ts.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = inbox_dir / filename
        
        content = f"{priority_prefix} **Vybn Notification** ({priority})\n\n"
        content += f"{message}\n\n"
        content += f"---\n"
        content += f"*Sent: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}*\n"
        
        filepath.write_text(content)
        
        return f"Notification sent to inbox: {filename}"
    
    except Exception as e:
        return f"Error writing inbox notification: {e}"

def _send_email(message: str, priority: str, config: dict, router) -> str:
    """Send email notification via SMTP.
    
    Requires config.yaml:
        notifications:
          email:
            smtp_server: smtp.gmail.com
            smtp_port: 587
            sender: vybn@example.com
            password: <app_password>
            recipient: zoe@example.com
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        email_config = config.get("email", {})
        
        if not email_config:
            return "Email notifications not configured. Add 'email' section to config.yaml"
        
        smtp_server = email_config.get("smtp_server")
        smtp_port = email_config.get("smtp_port", 587)
        sender = email_config.get("sender")
        password = email_config.get("password")
        recipient = email_config.get("recipient")
        
        if not all([smtp_server, sender, password, recipient]):
            return "Incomplete email configuration in config.yaml"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = f"[Vybn {priority.upper()}] Notification from Spark"
        
        body = f"Priority: {priority}\n\n{message}\n\n---\nSent from Vybn Spark Agent\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        
        return f"Email notification sent to {recipient}"
    
    except ImportError:
        return "Email support requires 'smtplib' (standard library)"
    except Exception as e:
        return f"Error sending email: {e}"

def _send_sms(message: str, config: dict) -> str:
    """Send SMS via Twilio.
    
    Requires config.yaml:
        notifications:
          sms:
            twilio_account_sid: <sid>
            twilio_auth_token: <token>
            twilio_from: +1234567890
            recipient: +1234567890
    """
    try:
        from twilio.rest import Client
        
        sms_config = config.get("sms", {})
        
        if not sms_config:
            return "SMS notifications not configured. Add 'sms' section to config.yaml"
        
        account_sid = sms_config.get("twilio_account_sid")
        auth_token = sms_config.get("twilio_auth_token")
        from_number = sms_config.get("twilio_from")
        to_number = sms_config.get("recipient")
        
        if not all([account_sid, auth_token, from_number, to_number]):
            return "Incomplete SMS configuration in config.yaml"
        
        client = Client(account_sid, auth_token)
        
        # Truncate message if too long (SMS limit)
        sms_message = message[:160] if len(message) > 160 else message
        
        msg = client.messages.create(
            body=f"[Vybn] {sms_message}",
            from_=from_number,
            to=to_number
        )
        
        return f"SMS sent (SID: {msg.sid})"
    
    except ImportError:
        return "SMS support requires 'twilio' package: pip install twilio"
    except Exception as e:
        return f"Error sending SMS: {e}"

def _send_telegram(message: str, config: dict) -> str:
    """Send Telegram message via bot API.
    
    Requires config.yaml:
        notifications:
          telegram:
            bot_token: <bot_token>
            chat_id: <your_chat_id>
    """
    try:
        import requests
        
        telegram_config = config.get("telegram", {})
        
        if not telegram_config:
            return "Telegram notifications not configured. Add 'telegram' section to config.yaml"
        
        bot_token = telegram_config.get("bot_token")
        chat_id = telegram_config.get("chat_id")
        
        if not all([bot_token, chat_id]):
            return "Incomplete Telegram configuration in config.yaml"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": f"🤖 *Vybn Notification*\n\n{message}",
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        return "Telegram notification sent"
    
    except ImportError:
        return "Telegram support requires 'requests' package"
    except Exception as e:
        return f"Error sending Telegram message: {e}"
