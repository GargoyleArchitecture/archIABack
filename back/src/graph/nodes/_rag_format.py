# -*- coding: utf-8 -*-
from pathlib import Path


def _safe_doc_label(metadata: dict) -> str:
    """Return a display-safe source label with no filesystem paths.

    Priority: source_title > title > basename (stem) of source_path/source.
    Never exposes absolute paths or OS usernames.
    """
    title = (metadata.get("source_title") or metadata.get("title") or "").strip()
    if title:
        return title
    raw = metadata.get("source_path") or metadata.get("source") or ""
    return Path(raw).stem or "doc"
