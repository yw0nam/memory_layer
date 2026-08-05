"""Unit tests for the killable MarkItDown conversion worker's failure reporting."""

from __future__ import annotations

from memory_base.adapters import document_worker


def test_main_prints_traceback_to_stderr_on_conversion_failure(monkeypatch, tmp_path, capsys):
    def failing_convert(input_path, output_path):
        raise ValueError("boom")

    monkeypatch.setattr(document_worker, "convert", failing_convert)
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.md"

    exit_code = document_worker.main([str(input_path), str(output_path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Traceback (most recent call last)" in stderr
    assert "ValueError: boom" in stderr
