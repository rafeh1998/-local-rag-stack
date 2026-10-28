"""Split documents into overlapping chunks for embedding."""


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into chunks of roughly `size` characters.

    Paragraphs are kept together where possible; a paragraph longer than
    `size` is split on character boundaries. Consecutive chunks share
    `overlap` characters so that sentences cut at a boundary still appear
    whole in at least one chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size - 1")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        # Paragraph may itself be too long: hard-split it.
        while len(current) + len(para) > size:
            room = size - len(current)
            chunks.append((current + para[:room]).strip())
            # Always advance by at least one character to guarantee progress.
            para = para[max(room - overlap, 1) :]
            current = ""
        current = f"{current}{para}" if current else para

    if current.strip():
        chunks.append(current.strip())
    return chunks
