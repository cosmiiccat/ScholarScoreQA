# fid/chunker.py

class PassageChunker:
    """
    Token-based passage chunking for long-context FiD QA.
    """

    def __init__(self, tokenizer, chunk_size=200, overlap=32):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, passage: str):
        token_ids = self.tokenizer.encode(passage, add_special_tokens=False)
        chunks = []

        step = self.chunk_size - self.overlap
        for i in range(0, len(token_ids), step):
            segment = token_ids[i : i + self.chunk_size]
            text = self.tokenizer.decode(segment, clean_up_tokenization_spaces=True)
            chunks.append(text)

            if i + self.chunk_size >= len(token_ids):
                break

        return chunks
