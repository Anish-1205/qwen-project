from __future__ import annotations

import logging

from logging_utils import DebugLogFormatter


def test_debug_formatter_makes_tagged_json_event_readable():
    record = logging.LogRecord(
        "chatbot_debug",
        logging.INFO,
        __file__,
        1,
        '[Turn 7] [Tool Result] {"tool":"calculator","ok":false,"error":{"code":"validation_error"}}',
        (),
        None,
    )

    rendered = DebugLogFormatter().format(record)

    assert "INFO" in rendered
    assert "TURN 7 · Tool Result" in rendered
    assert '\n    {' in rendered
    assert '      "code": "validation_error"' in rendered
    assert "\x1b[" not in rendered


def test_debug_formatter_includes_exception_traceback():
    try:
        raise ValueError("specific failure")
    except ValueError:
        record = logging.LogRecord(
            "chatbot_debug", logging.ERROR, __file__, 1, "Model load failed", (), __import__("sys").exc_info()
        )

    rendered = DebugLogFormatter().format(record)

    assert "ValueError: specific failure" in rendered
    assert "Model load failed" in rendered
