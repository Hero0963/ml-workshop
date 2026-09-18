# ai-collab/reports/artifacts/wrap-up-acceptance/gguf_compare.py
"""Are two GGUF files the same model? Compares metadata, the tensor table and every
tensor's bytes -- not the file hash, because Ollama re-orders tensors when it imports a
GGUF, so the served blob and the exported file hash differently (2026-09-19: 35/35
metadata keys and 426/426 tensors identical, order different).

Standard library only, so it runs in a throwaway container next to the Ollama volume:

    docker run --rm -v linkedin-zip-challenge_ollama_data:/o:ro -v <gguf dir>:/g:ro \
        -v <this dir>:/s:ro python:3.11-slim-trixie python /s/gguf_compare.py \
        /g/zip-qwen35-4b-p4c-text-f16.gguf /o/models/blobs/sha256-<digest>
"""

import hashlib
import struct
import sys
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
GGUF_STRING = 8
GGUF_ARRAY = 9
SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
# Bytes per element of the unquantised ggml types; the exported model has no others.
ELEMENT_BYTES = {0: 4, 1: 2, 30: 2}  # F32, F16, BF16

Tensor = tuple[str, tuple[int, ...], int, int]  # name, dims, ggml type, data offset


def _read(stream: BinaryIO, fmt: str) -> int | float | bool:
    return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))[0]


def _read_string(stream: BinaryIO) -> str:
    return stream.read(_read(stream, "Q")).decode("utf-8", "replace")


def _read_value(stream: BinaryIO, kind: int) -> object:
    if kind == GGUF_STRING:
        return _read_string(stream)
    if kind == GGUF_ARRAY:
        element_kind = _read(stream, "I")
        return [_read_value(stream, element_kind) for _ in range(_read(stream, "Q"))]
    return _read(stream, SCALAR_FORMATS[kind])


def read_header(path: Path) -> tuple[dict[str, object], list[Tensor], int]:
    """Metadata, tensor table, and the byte offset where tensor data may begin."""
    with path.open("rb") as stream:
        if stream.read(4) != GGUF_MAGIC:
            raise SystemExit(f"{path} is not a GGUF file")
        _read(stream, "I")  # version
        tensor_count, metadata_count = _read(stream, "Q"), _read(stream, "Q")
        metadata = {}
        for _ in range(metadata_count):
            key = _read_string(stream)
            metadata[key] = _read_value(stream, _read(stream, "I"))
        tensors = []
        for _ in range(tensor_count):
            name = _read_string(stream)
            dims = tuple(_read(stream, "Q") for _ in range(_read(stream, "I")))
            tensors.append((name, dims, _read(stream, "I"), _read(stream, "Q")))
        return metadata, tensors, stream.tell()


def tensor_digests(path: Path) -> tuple[dict[str, object], dict[str, str]]:
    metadata, tensors, header_end = read_header(path)
    alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    data_start = (header_end + alignment - 1) // alignment * alignment
    digests = {}
    with path.open("rb") as stream:
        for name, dims, kind, offset in tensors:
            if kind not in ELEMENT_BYTES:
                raise SystemExit(f"{name}: ggml type {kind} is not handled")
            size = ELEMENT_BYTES[kind]
            for dim in dims:
                size *= dim
            stream.seek(data_start + offset)
            digests[name] = hashlib.sha256(stream.read(size)).hexdigest()
    return metadata, digests


def main() -> None:
    first, second = Path(sys.argv[1]), Path(sys.argv[2])
    first_meta, first_tensors = tensor_digests(first)
    second_meta, second_tensors = tensor_digests(second)
    same_meta = sum(
        first_meta.get(key) == second_meta.get(key)
        for key in first_meta.keys() | second_meta.keys()
    )
    same_tensors = sum(
        first_tensors.get(name) == second_tensors.get(name)
        for name in first_tensors.keys() | second_tensors.keys()
    )
    print(f"metadata keys identical: {same_meta}/{len(first_meta | second_meta)}")
    print(
        f"tensor data identical:   {same_tensors}/{len(first_tensors | second_tensors)}"
    )
    print(f"same tensor order:       {list(first_tensors) == list(second_tensors)}")


if __name__ == "__main__":
    main()
