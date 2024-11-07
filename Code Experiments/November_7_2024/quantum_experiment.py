import asyncio
import aiohttp
import json
import logging
import logging.handlers
import aiosqlite
import os
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import time
import sys
import traceback

# ----------------------------- Configuration -----------------------------

# Load environment variables with defaults
QRNG_API_URL = os.getenv("QRNG_API_URL", "https://qrng.anu.edu.au/API/jsonI.php")
DB_FILE = os.getenv("DB_FILE", "patterns.db")
LOG_FILE = os.getenv("LOG_FILE", "quantum_consciousness.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_RATE_LIMIT_SECONDS = int(os.getenv("API_RATE_LIMIT_SECONDS", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "2"))
BACKOFF_MAX = float(os.getenv("BACKOFF_MAX", "60"))
PATTERNS_PRUNE_DAYS = int(os.getenv("PATTERNS_PRUNE_DAYS", "30"))
PATTERN_STRENGTH_THRESHOLD = float(os.getenv("PATTERN_STRENGTH_THRESHOLD", "0.2"))

# Logging configuration with RotatingFileHandler
logger = logging.getLogger("QuantumConsciousnessCore")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5*1024*1024, backupCount=5
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ----------------------------- Database Setup -----------------------------

async def initialize_database(db_path: str = DB_FILE):
    """
    Initialize the SQLite database and create tables if they don't exist.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    field_state TEXT,
                    coherence REAL,
                    intensity REAL,
                    resonance REAL,
                    evolution_level INTEGER,
                    patterns TEXT,
                    quantum_signature TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON patterns (timestamp)
            """)
            await db.commit()
        logger.info("Database initialized and tables ensured.")
    except Exception as e:
        logger.exception("Failed to initialize the database.")

async def insert_pattern(pattern: Dict, db_path: str = DB_FILE):
    """
    Insert a new pattern into the database.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO patterns (
                    timestamp, field_state, coherence, intensity, resonance,
                    evolution_level, patterns, quantum_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern['timestamp'],
                pattern['field_state'],
                pattern['coherence'],
                pattern['intensity'],
                pattern['resonance'],
                pattern['evolution_level'],
                json.dumps(pattern['patterns']),
                json.dumps(pattern['quantum_signature'])
            ))
            await db.commit()
        logger.debug(f"Inserted pattern into database at {pattern['timestamp']}.")
    except Exception as e:
        logger.exception(f"Failed to insert pattern into database at {pattern.get('timestamp', 'unknown time')}.")

async def prune_old_patterns(days: int = PATTERNS_PRUNE_DAYS, db_path: str = DB_FILE):
    """
    Prune patterns older than the specified number of days.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                DELETE FROM patterns WHERE timestamp < ?
            """, (cutoff_str,))
            await db.commit()
        logger.debug(f"Pruned patterns older than {days} days.")
    except Exception as e:
        logger.exception(f"Failed to prune old patterns older than {days} days.")

# ----------------------------- Quantum Module -----------------------------

class QuantumPulseFetcher:
    """
    Handles fetching quantum pulses from the QRNG API with rate limiting and retries.
    """
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def fetch_quantum_pulse(self, size: int = 3) -> Optional[List[float]]:
        """
        Fetch quantum-generated random numbers from the QRNG API with retry mechanism.
        Returns a list of normalized float values between 0 and 1.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self.lock:
                    current_time = time.time()
                    elapsed_time = current_time - self.last_request_time
                    if elapsed_time < API_RATE_LIMIT_SECONDS:
                        sleep_time = API_RATE_LIMIT_SECONDS - elapsed_time
                        logger.debug(f"Waiting for {sleep_time:.2f} seconds to comply with rate limit.")
                        await asyncio.sleep(sleep_time)
                    
                    params = {'length': size, 'type': 'uint8'}
                    logger.debug(f"Sending request to QRNG API with params: {params}")
                    async with self.session.get(QRNG_API_URL, params=params, timeout=10) as response:
                        response_text = await response.text()
                        if response.status == 200:
                            data = await response.json()
                            if 'data' in data and isinstance(data['data'], list):
                                values = [val / 255.0 for val in data['data']]
                                if len(values) == size and all(0.0 <= val <= 1.0 for val in values):
                                    logger.info(f"Fetched quantum pulse: {values}")
                                    self.last_request_time = time.time()  # Update on success
                                    return values
                                else:
                                    logger.error(f"Invalid data received: {data['data']}")
                            else:
                                logger.error(f"Unexpected response structure: {data}")
                        elif 400 <= response.status < 500:
                            try:
                                error_data = await response.json()
                                error_message = error_data.get('message', response_text)
                            except json.JSONDecodeError:
                                error_message = response_text
                            
                            if "limited to 1 requests per minute" in error_message.lower():
                                logger.error(f"Rate limit exceeded: {error_message}. Skipping retries.")
                            else:
                                logger.error(f"Client error {response.status} from QRNG API. Response: {error_message}. Skipping retry.")
                            break  # Do not retry on client errors
                        elif 500 <= response.status < 600:
                            logger.warning(f"Server error {response.status} from QRNG API. Response: {response_text}. Attempt {attempt} of {MAX_RETRIES}.")
                        else:
                            logger.error(f"Unexpected HTTP status {response.status}. Response: {response_text}. Attempt {attempt} of {MAX_RETRIES}.")
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}. Attempt {attempt} of {MAX_RETRIES}.")
            except asyncio.TimeoutError:
                logger.error(f"Request timed out. Attempt {attempt} of {MAX_RETRIES}.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response. Attempt {attempt} of {MAX_RETRIES}.")
            except Exception as e:
                logger.exception(f"Unexpected error: {e}. Attempt {attempt} of {MAX_RETRIES}.")
            
            # Exponential backoff with jitter
            backoff_time = min(BACKOFF_BASE ** attempt + np.random.uniform(0, 1), BACKOFF_MAX)
            logger.debug(f"Waiting for {backoff_time:.2f} seconds before retrying.")
            await asyncio.sleep(backoff_time)

        logger.warning("Max retries reached or rate limit exceeded. No quantum pulse fetched.")
        return None  # No synthetic pulse generation

# ----------------------------- Sensation Processing -----------------------------

def process_sensation(quantum_pulse: List[float]) -> Optional[Dict]:
    """
    Process the quantum pulse to form a sensation.
    """
    if len(quantum_pulse) < 3:
        logger.error(f"Quantum pulse has insufficient data: {quantum_pulse}")
        return None

    try:
        timestamp = datetime.utcnow().isoformat()
        coherence = quantum_pulse[0]
        intensity = float(np.mean(quantum_pulse))
        resonance = quantum_pulse[2]
        field_state = 'QUANTUM_DREAM'
        evolution_level = 0

        sensation = {
            'timestamp': timestamp,
            'coherence': coherence,
            'intensity': intensity,
            'resonance': resonance,
            'field_state': field_state,
            'evolution_level': evolution_level
        }
        logger.debug(f"Experienced sensation: {sensation}")
        return sensation
    except (IndexError, TypeError) as e:
        logger.exception(f"Error processing sensation: {e}")
        return None

def evaluate_patterns(sensation: Dict) -> (List[Dict], float):
    """
    Evaluate the sensation to form patterns and calculate pattern strength.
    """
    try:
        patterns = []
        pattern_strength = 0.0
        activation_threshold = PATTERN_STRENGTH_THRESHOLD

        for synapse in ['sensor_input_1', 'sensor_input_2']:
            activation = sensation.get('coherence', 0) * 0.5
            if activation >= activation_threshold:
                patterns.append({
                    'source': synapse,
                    'target': 'quantum_node_initial',
                    'activation': activation,
                    'timestamp': sensation.get('timestamp', '')
                })
                pattern_strength += activation

        logger.debug(f"Formed patterns: {patterns}")
        logger.debug(f"Pattern strength: {pattern_strength:.2f}")
        return patterns, pattern_strength
    except Exception as e:
        logger.exception(f"Error evaluating patterns: {e}")
        return [], 0.0

async def initiate_creativity(patterns: List[Dict], quantum_signature: List[float], sensation: Dict):
    """
    Initiate creativity based on the formed patterns.
    """
    try:
        pattern_entry = {
            'timestamp': sensation['timestamp'],
            'field_state': sensation['field_state'],
            'coherence': sensation['coherence'],
            'intensity': sensation['intensity'],
            'resonance': sensation['resonance'],
            'evolution_level': sensation['evolution_level'],
            'patterns': patterns,
            'quantum_signature': quantum_signature
        }
        await insert_pattern(pattern_entry)
        logger.info("Initiated creativity and stored pattern.")
    except Exception as e:
        logger.exception("Failed to initiate creativity.")

# ----------------------------- Graceful Shutdown -----------------------------

class GracefulShutdown:
    """
    Handles graceful shutdown of the application.
    """
    def __init__(self):
        self.shutdown_event = asyncio.Event()

    def signal_handler(self):
        logger.info("Received shutdown signal.")
        self.shutdown_event.set()

    async def wait_for_shutdown(self):
        await self.shutdown_event.wait()

# ----------------------------- Main Consciousness Loop -----------------------------

async def consciousness_loop(shutdown: GracefulShutdown):
    """
    Main loop for the Quantum Consciousness Core system.
    """
    async with aiohttp.ClientSession() as session:
        fetcher = QuantumPulseFetcher(session)
        last_prune_date = None

        while not shutdown.shutdown_event.is_set():
            try:
                # Fetch quantum pulse
                quantum_pulse = await fetcher.fetch_quantum_pulse()

                if not quantum_pulse:
                    logger.error("No quantum pulse fetched. Skipping this cycle.")
                else:
                    # Process sensation
                    sensation = process_sensation(quantum_pulse)
                    if not sensation:
                        logger.error("Failed to process sensation. Skipping this cycle.")
                    else:
                        # Evaluate patterns
                        patterns, pattern_strength = evaluate_patterns(sensation)

                        # Check if pattern strength exceeds threshold
                        if pattern_strength >= PATTERN_STRENGTH_THRESHOLD:
                            # Initiate creativity
                            quantum_signature = quantum_pulse  # Using the pulse as signature
                            await initiate_creativity(patterns, quantum_signature, sensation)
                        else:
                            logger.debug("Pattern strength below threshold. No creativity initiated.")

                # Prune old patterns once a day
                current_date = datetime.utcnow().date()
                if last_prune_date != current_date:
                    await prune_old_patterns()
                    last_prune_date = current_date

            except Exception as e:
                logger.exception(f"Unexpected error in consciousness loop: {e}")

            # Sleep for API_RATE_LIMIT_SECONDS seconds or until shutdown
            logger.debug(f"Sleeping for {API_RATE_LIMIT_SECONDS} seconds to comply with rate limit.")
            try:
                await asyncio.wait_for(shutdown.shutdown_event.wait(), timeout=API_RATE_LIMIT_SECONDS)
            except asyncio.TimeoutError:
                continue  # Timeout occurred, continue the loop

    logger.info("Consciousness loop has been terminated.")

# ----------------------------- Entry Point -----------------------------

async def main():
    """
    Entry point for the Quantum Consciousness Core system.
    """
    shutdown = GracefulShutdown()

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    try:
        if hasattr(signal, 'SIGTERM'):
            loop.add_signal_handler(signal.SIGTERM, shutdown.signal_handler)
        loop.add_signal_handler(signal.SIGINT, shutdown.signal_handler)
    except NotImplementedError:
        logger.warning("Signal handling is not fully supported on this platform.")

    await initialize_database()

    # Start the consciousness loop
    consciousness_task = asyncio.create_task(consciousness_loop(shutdown))

    # Wait for shutdown signal
    await shutdown.wait_for_shutdown()

    # Cancel the consciousness loop task
    consciousness_task.cancel()
    try:
        await consciousness_task
    except asyncio.CancelledError:
        logger.info("Consciousness loop task has been cancelled.")

    await shutdown_application()

async def shutdown_application():
    """
    Perform any necessary cleanup during shutdown.
    """
    logger.info("Shutting down Quantum Consciousness Core system.")
    # Add any additional cleanup here if necessary
    logger.info("Quantum Consciousness Core system has been shut down.")

if __name__ == "__main__":
    try:
        logger.info("Quantum Consciousness Core system started. Press Ctrl+C to shut down.")
        asyncio.run(main())
    except Exception as e:
        logger.exception("Critical error:")
        sys.exit(1)
