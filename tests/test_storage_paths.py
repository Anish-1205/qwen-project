from __future__ import annotations

import inspect
import logging
import tempfile
import unittest
from pathlib import Path

from app_paths import (
    AGENT_MEMORY_DB_PATH,
    CHAT_SESSIONS_DB_PATH,
    DATA_DIR,
    DEBUG_LOG_PATH,
    DOCUMENT_DB_PATH,
    PROJECT_ROOT,
)
from documents.config import DOCUMENT_DB_PATH as DOCUMENT_CONFIG_DB_PATH
from documents.db import init_db
from logging_utils import DEFAULT_LOG_PATH, setup_debug_logger
from memory_core import OfflineMemoryManager


class StoragePathTests(unittest.TestCase):
    def test_runtime_defaults_share_the_data_directory(self):
        expected = {
            "agent_memory.db": AGENT_MEMORY_DB_PATH,
            "chat_sessions.db": CHAT_SESSIONS_DB_PATH,
            "documents.db": DOCUMENT_DB_PATH,
            "chatbot_debug.log": DEBUG_LOG_PATH,
        }
        for filename, path in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(path, DATA_DIR / filename)

        self.assertEqual(DATA_DIR, PROJECT_ROOT / "data")
        self.assertEqual(DOCUMENT_CONFIG_DB_PATH, DOCUMENT_DB_PATH)
        self.assertEqual(DEFAULT_LOG_PATH, DEBUG_LOG_PATH)
        memory_default = inspect.signature(OfflineMemoryManager).parameters["db_path"].default
        self.assertEqual(memory_default, AGENT_MEMORY_DB_PATH)
        webapp_source = (PROJECT_ROOT / "webapp.py").read_text(encoding="utf-8")
        self.assertIn("from app_paths import CHAT_SESSIONS_DB_PATH", webapp_source)

    def test_database_and_log_helpers_create_nested_parents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "nested" / "data"
            document_db = root / "documents.db"
            memory_db = root / "agent_memory.db"
            log_file = root / "chatbot_debug.log"

            init_db(document_db)
            OfflineMemoryManager(memory_db, embed_model=object())
            logger, resolved_log = setup_debug_logger(log_file)
            try:
                self.assertTrue(document_db.is_file())
                self.assertTrue(memory_db.is_file())
                self.assertEqual(resolved_log, log_file.resolve())
                self.assertTrue(log_file.is_file())
            finally:
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()
                logging.shutdown()


if __name__ == "__main__":
    unittest.main()
