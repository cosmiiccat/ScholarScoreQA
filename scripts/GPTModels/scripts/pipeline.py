# pipeline/simple_prompt_pipeline.py

from .text_utils import clean
from .prompt_utils import build_prompt
from .model import OpenAIPromptModel

class SimplePromptPipeline:

    def run(self, context: str, question: str):
        context = clean(context)
        question = clean(question)

        prompt = build_prompt(context, question)
        answer = OpenAIPromptModel.query_model(prompt)

        return {
            "context": context,
            "question": question,
            "answer": answer
        }
