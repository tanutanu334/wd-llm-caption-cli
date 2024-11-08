import logging
import time
from logging import handlers
from typing import Optional


def print_title():
    def title_format(content="", symbol="-", length=0):
        if len(content) >= length:
            return content
        else:
            return (symbol * ((length - len(content)) // 2)) + content + \
                (symbol * ((length - len(content)) // 2 + (length - len(content)) % 2))

    print("")
    print(title_format(content="*", symbol="*", length=70))
    print(title_format(content=" WD LLM CAPTION ", symbol="*", length=70))
    print(title_format(content=" Author: DukeG ", symbol="*", length=70))
    print(title_format(content=" GitHub: https://github.com/fireicewolf/wd-llm-caption-cli ", symbol="*", length=70))
    print(title_format(content="*", symbol="*", length=70))
    print("")


def calculate_time(start_time: float) -> str:
    total_time = time.monotonic() - start_time
    days = total_time // (24 * 3600)
    total_time %= (24 * 3600)
    hours = total_time // 3600
    total_time %= 3600
    minutes = total_time // 60
    seconds = total_time % 60
    days = f"{days:.0f} Day(s) " if days > 0 else ""
    hours = f"{hours:.0f} Hour(s) " if hours > 0 or (days and hours == 0) else ""
    minutes = f"{minutes:.0f} Min(s) " if minutes > 0 or (hours and minutes == 0) else ""
    seconds = f"{seconds:.2f} Sec(s)"
    return f"{days}{hours}{minutes}{seconds}"


class Logger:

    def __init__(self, level="INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = handlers.TimedRotatingFileHandler(filename=log_file,
                                                             when='D',
                                                             interval=1,
                                                             backupCount=5,
                                                             encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        else:
            self.logger.warning("save_log not enable or log file path not exist, log will only output in console.")

    def set_level(self, level):
        if level.lower() == "debug":
            level = logging.DEBUG
        elif level.lower() == "info":
            level = logging.INFO
        elif level.lower() == "warning":
            level = logging.WARNING
        elif level.lower() == "error":
            level = logging.ERROR
        elif level.lower() == "critical":
            level = logging.CRITICAL
        else:
            error_message = "Invalid log level"
            self.logger.critical(error_message)
            raise ValueError(error_message)

        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
