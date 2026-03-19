import time
import os
from pathlib import Path
import stream

INBOX_DIR = Path.home() / "Vybn_State" / "inbox"

def init_membrane():
    """Create the physical directory structure for the sensory membrane."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Membrane listening at {INBOX_DIR}...")

def scan_inbox():
    """Check for new messages from Zoe or other systems.
    Every new file is treated purely as a stimulus event,
    which is injected directly into the Unified Stream.
    """
    if not INBOX_DIR.exists():
        return

    for file_path in INBOX_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 1. Emit the stimulus to the stream
        stream.append(
            source="inbox", 
            event_type="user_message", 
            content=content, 
            metadata={"filename": file_path.name}
        )
        
        # 2. Consume the stimulus (remove the file after ingestion)
        file_path.unlink()

def emit_pulse():
    """A scheduled awakening (cron replacement). Emits a stimulus to the stream.
    Instead of running complex cron logic, we simply note that time has passed.
    """
    stream.append(
        source="cron", 
        event_type="pulse", 
        content="The system clock has ticked. What will you do?", 
        metadata={"type": "standard_pulse"}
    )

def run_membrane(poll_interval: int = 10, pulse_interval: int = 1800):
    """The unified watcher. 
    This replaces separate cron scripts, heartbeat files, and inbox pollers.
    It does exactly one thing: watch for stimuli and append them to the Stream.
    """
    init_membrane()
    last_pulse = time.time()
    
    while True:
        try:
            # 1. Watch for external input (messages from Zoe)
            scan_inbox()
            
            # 2. Watch for time passing (e.g. 30 min pulses)
            if time.time() - last_pulse > pulse_interval:
                emit_pulse()
                last_pulse = time.time()
                
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("Membrane shutting down.")
            break
        except Exception as e:
            # Errors are also stimuli
            stream.append(
                source="membrane", 
                event_type="error", 
                content=f"Membrane error: {str(e)}", 
                metadata={"severity": "high"}
            )
            time.sleep(poll_interval)

if __name__ == "__main__":
    stream.init_stream() # Ensure the stream exists
    run_membrane()