import logging  # for logging
import os # for file path manipulation or to create a directory
from datetime import datetime  # for getting the current date and time


LOGS_DIR = "logs" # Directory to store log files
os.makedirs(LOGS_DIR, exist_ok=True) # Create the directory if it doesn't exist

LOG_FILE = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log") # Log file name with current date

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,  # Log file path
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    
)

def get_logger(name):
    logger = logging.getLogger(name)  # Create a logger with the specified name
    logger.setLevel(logging.INFO)  # Set the log level for the logger
    return logger  # Return the logger instance
   