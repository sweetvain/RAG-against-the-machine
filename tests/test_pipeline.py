from pathlib import Path
from typing import List

import pytest

from src.evaluation import compute_question_recall
from src.ingestion import (
    chunk_section,
    run_ingestion,
    tokenize_text,
)
from src.models import MinimalSource
from src.retrieval import load_index, search


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _check_chunks(path: Path, size: int) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    chunks = chunk_section(path, max_chunk_size=size)
    assert chunks
    for chunk in chunks:
        meta = chunk["metadata"]
        start = meta["first_character_index"]
        end = meta["last_character_index"]
        assert end - start + 1 <= size
        assert text[start:end + 1] == chunk["content"]


def test_python_chunks_respect_size_and_offsets(tmp_path: Path) -> None:
    code = "import os\n\n" + "".join(
        f"@deco\ndef f{i}(x):\n    return x + {i}\n\n" for i in range(50)
    )
    code += "x = '" + "a" * 5000 + "'\n"
    _check_chunks(_write(tmp_path / "a.py", code), 300)


def test_markdown_chunks_ignore_headings_in_code_fences(
    tmp_path: Path,
) -> None:
    md = "# Title\ntext\n```\n# not a heading\n```\n## Next\nmore\n"
    path = _write(tmp_path / "a.md", md)
    chunks = chunk_section(path, max_chunk_size=2000)
    assert len(chunks) == 2
    _check_chunks(path, 20)


def test_crlf_offsets_are_exact(tmp_path: Path) -> None:
    md = "# A\r\nline\r\n# B\r\nline\r\n"
    _check_chunks(_write(tmp_path / "crlf.md", md), 2000)


def test_empty_file_gives_no_chunk(tmp_path: Path) -> None:
    assert chunk_section(_write(tmp_path / "e.py", ""), 100) == []


def test_invalid_chunk_size_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.py", "def f():\n    pass\n")
    with pytest.raises(ValueError):
        chunk_section(path, max_chunk_size=0)


def test_tokenizer_keeps_identifiers_and_parts() -> None:
    tokens: List[str] = tokenize_text("load_lora_adapter")
    assert "load_lora_adapter" in tokens
    assert "lora" in tokens


def test_index_and_search(tmp_path: Path) -> None:
    corpus = tmp_path / "raw"
    corpus.mkdir()
    _write(corpus / "a.md", "# LoRA\nload adapter with lora endpoint\n")
    _write(corpus / "b.py", "def other():\n    return 'unrelated'\n")
    out = tmp_path / "processed"
    run_ingestion(str(corpus), str(out), max_chunk_size=2000)
    retriever, meta = load_index(str(out))

    assert search("lora adapter", retriever, meta, 1)[0].file_path.endswith(
        "a.md"
    )
    assert search("", retriever, meta, 3) == []
    assert search("lora", retriever, meta, 0) == []
    assert search("lora", retriever, meta, -2) == []
    assert search("!!! ???", retriever, meta, 3) == []


def test_recall_uses_iou_threshold() -> None:
    ref = MinimalSource(
        file_path="a", first_character_index=0, last_character_index=99
    )
    hit = MinimalSource(
        file_path="a", first_character_index=50, last_character_index=150
    )
    miss = MinimalSource(
        file_path="b", first_character_index=0, last_character_index=99
    )
    assert compute_question_recall([ref], [hit], 1) == 1.0
    assert compute_question_recall([ref], [miss], 1) == 0.0
    assert compute_question_recall([ref], [hit], 0) == 0.0
