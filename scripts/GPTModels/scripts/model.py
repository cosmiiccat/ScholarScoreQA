# models/openai_prompt_model.py

from config import OPENAI_CLIENT, MODEL_NAME

class OpenAIPromptModel:

    @staticmethod
    def query_model(prompt: str) -> str:
        """
        Generic function to send a prompt to OpenAI.
        """
        response = OPENAI_CLIENT.responses.create(
            model=MODEL_NAME,
            input=prompt
        )

        return response.output[0].content[0].text
