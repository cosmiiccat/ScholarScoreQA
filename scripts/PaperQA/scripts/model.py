# paperqa_core/model.py

import os
import tempfile
import asyncio
import concurrent.futures
from paperqa import Docs

from paperqa_core.config import PaperQAConfig


class PaperQAModel:
    """
    Wrapper around PaperQA Docs() with synchronous execution
    inside a thread executor, using the user's original logic.
    """

    def __init__(self):
        # Initialize Docs object
        self.docs = Docs()
        # Initialize default question
        self.question = PaperQAConfig.DEFAULT_QUESTION

        # Ensure OpenAI key is set
        if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "":
            raise ValueError("OPENAI_API_KEY is not set. Use set_openai_key().")

    def answer(self, prompt: str, question: str) -> str:
        """
        Takes a text prompt and a question to ask PaperQA.
        Creates a temp file, adds it to Docs, and queries the model.
        """

        self.question = question

        def add_and_query_sync():
            # Create a temporary file with passage text
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(prompt)
                tmp_file_path = tmp_file.name

            try:
                # Add to PaperQA
                self.docs.add(tmp_file_path, dockey="prompt_text")

                # Query PaperQA
                answer = self.docs.query(self.question)
                return answer.formatted_answer

            finally:
                # Clean temp file
                os.remove(tmp_file_path)

        # Threaded execution for sync blocking PaperQA calls
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(add_and_query_sync)
            result = future.result()

        return result
