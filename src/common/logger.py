import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from src.common.training_utils import get_path


class Logger:
    def __init__(
        self,
        working_dir,
        specific_excel_file_name=None,
        max_log_size_mb=10,
        backup_count=5,
        stream_to_stdout=True,
        log_level=logging.INFO,
        enable_logging=True,
        log_to_file=True,
        log_to_console=True,
    ):
        self.specific_excel_file_name = specific_excel_file_name
        self.max_log_size_mb = max_log_size_mb
        self.backup_count = backup_count
        self.stream_to_stdout = stream_to_stdout
        self.log_to_file = log_to_file
        self.enable_logging = enable_logging
        self.log_level = log_level
        self.log_to_console = log_to_console
        self.working_dir = working_dir
        if self.enable_logging:
            self.logger = logging.getLogger(__name__)
            self.configure_logging()
        else:
            self.logger = logging.getLogger()
            self.logger.disabled = True
            self.logger.propagate = False

    def configure_logging(self):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handlers = []
        config_info = []
        if (
            self.log_info_to_file(formatter, handlers, config_info)
            and self.specific_excel_file_name is not None
        ):
            self.log_info_to_file(formatter, handlers, config_info)
        if self.log_info_to_console(formatter, handlers, config_info):
            self.log_info_to_console(formatter, handlers, config_info)

        logging.basicConfig(level=self.log_level, handlers=handlers, force=True)
        self.logger.info(
            f"#Logging configured at level {logging.getLevelName(self.log_level)}"
        )

        for info in config_info:
            self.logger.info(info)

    def log_info_to_file(self, formatter, handlers, config_info):
        log_file = get_path(
            base_path=self.working_dir,
            filename=f"classifier_{datetime.now():%Y%m%d-%H%M%S}_{self.specific_excel_file_name}.log",
            create=True,
        )

        max_bytes = self.max_log_size_mb * 1024 * 1024
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)
        handlers.append(file_handler)
        config_info.append(
            f"File logging: {log_file} (max {self.max_log_size_mb}MB, {self.backup_count} backups)"
        )
        return True

    def log_info_to_console(self, formatter, handlers, config_info):
        stream = sys.stdout if self.stream_to_stdout else sys.stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)
        handlers.append(console_handler)
        stream_name = "stdout" if self.stream_to_stdout else "stderr"
        config_info.append(f"Console Logging: {stream_name}")
        return True
