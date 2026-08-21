from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from documents.db import init_db
from documents.index import DocumentIndex


class ConstantEmbedModel:
    def encode(self, text):
        if isinstance(text, list):
            return np.asarray([[1.0, 0.0] for _ in text], dtype=np.float32)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class DocumentFilenameRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "documents.db"
        self.docs_dir = Path(self.temp_dir.name) / "knowledge"
        self.docs_dir.mkdir()
        init_db(self.db_path)
        now = "2026-01-01 00:00:00"
        vector = np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            for doc_id, filename, text in [
                ("target", "upload_test.txt", "The secret code phrase is cedar moon."),
                ("decoy", "security_playbook.pdf", "Unrelated but semantically similar security instructions."),
            ]:
                conn.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, filename, str(self.docs_dir / filename), Path(filename).suffix, len(text), doc_id, now, now, "indexed"),
                )
                conn.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"chunk-{doc_id}", doc_id, 0, None, 1, 1, text, vector, now),
                )
        self.index = DocumentIndex(
            ConstantEmbedModel(),
            db_path=self.db_path,
            docs_dir=self.docs_dir,
            similarity_threshold=0.0,
            router_threshold=0.0,
        )

    def tearDown(self):
        del self.index
        gc.collect()
        self.temp_dir.cleanup()

    def test_explicit_filename_scopes_retrieval_to_that_document(self):
        result = self.index.lookup_context("What is the secret code phrase in upload_test.txt?")

        self.assertTrue(result.routed_relevant)
        self.assertEqual(result.retrieved_count, 1)
        self.assertIn("cedar moon", result.context)
        self.assertNotIn("security instructions", result.context)

    def test_missing_explicit_filename_does_not_inject_semantic_decoys(self):
        result = self.index.lookup_context("What is the secret code phrase in missing_file.txt?")

        self.assertFalse(result.routed_relevant)
        self.assertEqual(result.retrieved_count, 0)
        self.assertEqual(result.context, "")
        self.assertEqual(result.reason, "named document not indexed")


if __name__ == "__main__":
    unittest.main()
