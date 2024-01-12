from os import environ


class Log:
    log_level = environ.get("LOG_LEVEL", "INFO")

    @classmethod
    def __should_log(cls, level: str):
        app_level = Log.log_level or "INFO"
        level_map = {
            "TRACE": 5,
            "DEBUG": 4,
            "INFO": 3,
            "WARNING": 2,
            "ERROR": 1,
            "FATAL": 0,
        }
        return level_map.get(app_level, 2) >= level_map.get(level, 4)

    @classmethod
    def trace(cls, *args, **kwargs):
        if not Log.__should_log("TRACE"):
            return
        print("TRACE:", *args, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        if not Log.__should_log("DEBUG"):
            return
        print("DEBUG:", *args, **kwargs)

    @classmethod
    def info(cls, *args, **kwargs):
        if not Log.__should_log("INFO"):
            return
        print("INFO:", *args, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        if not Log.__should_log("WARNING"):
            return
        print("WARNING:", *args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        if not Log.__should_log("ERROR"):
            return
        print("ERROR:", *args, **kwargs)

    @classmethod
    def fatal(cls, *args, **kwargs):
        if not Log.__should_log("FATAL"):
            return
        print("FATAL:", *args, **kwargs)

    @classmethod
    def set_level(cls, level: str):
        if level in ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]:
            Log.log_level = level
        else:
            raise ValueError(f"Invalid log level: {level}")

    @classmethod
    def is_level_above(cls, lvl="INFO"):
        return cls.__should_log(lvl)
