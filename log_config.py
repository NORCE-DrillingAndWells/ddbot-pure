# log_config.py
import logging
import os


def configure_logger(log_file_path: str, mode: str = "w"):
    """
    config and save log to file
    :param log_file_path: log file path
    """

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode=mode, encoding="utf-8"), logging.StreamHandler()],
    )


def change_logfile_path(log_file_path: str, mode: str = "w"):
    """
    change log file
    :param log_file_path: log file path
    """

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode=mode, encoding="utf-8"), logging.StreamHandler()],
    )
    # logging.addHandler(logging.FileHandler(log_file_path))
    # logging.addHandler(logging.StreamHandler())
