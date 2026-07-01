import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(logfile_name, log_level=logging.INFO, log_dir="logs"):
    """
    Configure logging with console and rotating file handlers.
    Prevents adding multiple handlers to the same logger if called multiple times.
    """
    logger = logging.getLogger("ADV-DF-Logger")

    # Check if the logger already has handlers, which would suggest it's already configured
    if not logger.handlers:
        logger.setLevel(log_level)

        # Create logs directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a file handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, logfile_name),
            maxBytes=10**6,  # 1MB
            backupCount=5,
        )
        file_log_level = int(os.environ.get("LOG_LEVEL_FILE", log_level))
        file_handler.setLevel(file_log_level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Create formatters and add them to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def initialize_logger(args):
    """Initialize logger with names for the shipped attack/evaluation entry points."""
    detector = getattr(args, "detector_name", "unknown")
    dataset = getattr(args, "dataset_name", "unknown")

    if hasattr(args, "attack_type"):
        process_name = "attack"
        attack_type = getattr(args, "attack_type", "unknown")
        logfile_name = f"attack_{attack_type}_{detector}_{dataset}.log"
    else:
        process_name = "evaluation"
        logfile_name = f"evaluation_{detector}_{dataset}.log"

    log_dir = os.path.join(args.output_dir, "logs")
    logger = setup_logging(logfile_name, log_dir=log_dir)
    logger.info(f"Logger initialized. Starting the {process_name} process...")

    return logger
