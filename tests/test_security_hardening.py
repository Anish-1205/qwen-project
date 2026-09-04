from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from tools import config
from tools.common import ToolError, resolve_file
from tools.directory_listing import list_directory
from webapp import DOCUMENT_UPLOAD_MAX_BYTES, _save_uploaded_document


def test_local_tools_reject_paths_outside_allowed_roots(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    inside_file = allowed / "inside.txt"
    outside_file = outside / "outside.txt"
    inside_file.write_text("ok", encoding="utf-8")
    outside_file.write_text("secret", encoding="utf-8")

    with patch.object(config, "LOCAL_ALLOWED_ROOTS", (allowed.resolve(),)):
        assert resolve_file(str(inside_file), {".txt"}, 100) == inside_file.resolve()
        assert list_directory(str(allowed))["count"] == 1
        for path in (outside_file, allowed / ".." / "outside" / "outside.txt"):
            with pytest.raises(ToolError) as error:
                resolve_file(str(path), {".txt"}, 100)
            assert error.value.code == "path_outside_allowed_roots"
        with pytest.raises(ToolError):
            list_directory(str(outside))


def test_document_upload_validation_and_collision(tmp_path):
    with patch("webapp._get_documents_root", return_value=tmp_path):
        saved = _save_uploaded_document("note.txt", b"hello")
        assert saved.read_bytes() == b"hello"
        with pytest.raises(HTTPException) as collision:
            _save_uploaded_document("note.txt", b"replacement")
        assert collision.value.status_code == 409
        assert saved.read_bytes() == b"hello"
        with pytest.raises(HTTPException) as extension:
            _save_uploaded_document("payload.exe", b"x")
        assert extension.value.status_code == 415
        with pytest.raises(HTTPException) as oversized:
            _save_uploaded_document("large.pdf", b"x" * (DOCUMENT_UPLOAD_MAX_BYTES + 1))
        assert oversized.value.status_code == 413
