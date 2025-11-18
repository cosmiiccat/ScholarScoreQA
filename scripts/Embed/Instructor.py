# instructor_embeddings.py
import torch
from InstructorEmbedding import INSTRUCTOR
from typing import List


class InstructorEmbeddingModel:
    def __init__(self, model_name: str = "hkunlp/instructor-base"):
        """
        Common models:
        - hkunlp/instructor-base
        - hkunlp/instructor-large
        - hkunlp/instructor-xl
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = INSTRUCTOR(model_name).to(self.device)

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        INSTRUCTOR requires input in format:
        ["question", "text"]
        """
        instruct_pairs = [["Represent the text for retrieval:", text] for text in texts]

        with torch.no_grad():
            embeddings = self.model.encode(instruct_pairs)

        return torch.tensor(embeddings)


