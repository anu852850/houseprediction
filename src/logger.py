import logging
import os
from datetime import datetime

# STEP 1: Generate a unique log file name with the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# STEP 2: Create a 'logs' folder path in the current working directory
logs_folder = os.path.join(os.getcwd(), "logs")

# STEP 3: Make the logs folder (if it doesn't already exist)
os.makedirs(logs_folder, exist_ok=True)

# STEP 4: Combine folder + file name to get full path to the log file
LOG_FILE_PATH = os.path.join(logs_folder, LOG_FILE)

# STEP 5: Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
if __name__=="__main__":
    logging.info("logging has started")