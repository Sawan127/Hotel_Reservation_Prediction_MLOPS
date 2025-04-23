from src.logger import get_logger
from src.custom_exception import CustomException
import sys


logger = get_logger(__name__)

def divide(a, b):
    try:
        result = a / b
        logger.info(f"Division successful: {result}")
        return result
    except Exception as e:
        logger.error("Error occurred in divide function")
        raise CustomException("Division by zero is not allowed", sys)
    
if __name__ == "__main__":
    try:
        logger.info("Starting the division operation")
        divide(10, 2)
    except CustomException as ce:
        logger.error(str(ce))