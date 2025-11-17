import os
import openai
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ToneTransformer:
    def __init__(self):
        """Initialize the Tone Transformer with OpenAI API configuration"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        self.valid_tones = {
            "academic": "Academic (Scholars, researchers)",
            "technical": "Technical/Factual (Experts)",
            "descriptive": "Descriptive (Students, General readers)",
            "conversational": "Conversational",
            "simplified": "Simplified/Layman-Friendly"
        }

    def create_tone_prompt(self, answer: str, tone: str) -> str:
        """Create a detailed prompt for transforming the answer into the specified tone"""
        if tone.lower() not in self.valid_tones:
            raise ValueError(f"Invalid tone. Choose from: {', '.join(self.valid_tones.keys())}")

        tone_description = self.valid_tones[tone.lower()]
        prompt = """
You are an expert writer skilled in adapting text to various tones while preserving its core meaning. Your task is to transform the given answer into the specified tone, ensuring the factual content remains accurate and the style matches the target audience. Below are the tone definitions and the input answer.

Tone Definitions:
1. Academic (Scholars, researchers): Formal, precise, uses scholarly language, often includes technical terms and a structured approach.
2. Technical/Factual (Experts): Concise, fact-driven, uses industry-specific terminology, avoids embellishment.
3. Descriptive (Students, General readers): Detailed, illustrative, uses examples and clear explanations to enhance understanding.
4. Conversational: Casual, friendly, uses natural speech patterns as if speaking to a friend.
5. Simplified/Layman-Friendly: Very simple language, avoids jargon, explains concepts as if to someone with no prior knowledge.

Input Answer: {answer}

Requested Tone: {tone_description}

Instructions:
1. Rewrite the answer in the specified tone.
2. Preserve the original meaning and factual accuracy.
3. Adapt the vocabulary, sentence structure, and style to match the tone's target audience.
4. Return only the transformed answer as plain text (no additional formatting or explanation).

Please provide the transformed answer below:
""".format(
    answer=answer,
    tone_description=tone_description
)
        return prompt

    def transform_tone(self, answer: str, tone: str) -> str:
        """Transform the answer into the specified tone using GPT-4o-mini"""
        try:
            prompt = self.create_tone_prompt(answer, tone)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a skilled writer specializing in tone adaptation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly higher temperature for stylistic variation
                max_tokens=500
            )
            
            # Extract the transformed answer
            transformed_answer = response.choices[0].message.content.strip()
            return transformed_answer

        except Exception as e:
            return f"Error transforming tone: {str(e)}"

    def transform_all_tones(self, answer: str) -> dict:
        """Transform the answer into all available tones"""
        result = {}
        for tone in self.valid_tones.keys():
            result[tone] = self.transform_tone(answer, tone)
        return result

    def print_transformed_answer(self, answer: str, tone: str):
        """Print the transformed answer for a specific tone"""
        transformed = self.transform_tone(answer, tone)
        print(f"\nOriginal Answer: {answer}")
        print(f"Transformed to {self.valid_tones[tone.lower()]}:")
        print(transformed)

    def print_all_transformations(self, answer: str):
        """Print transformations for all tones"""
        results = self.transform_all_tones(answer)
        print(f"\nOriginal Answer: {answer}")
        print("\nTransformations:")
        for tone, transformed in results.items():
            print(f"{self.valid_tones[tone]}:")
            print(f"{transformed}\n")


# Example usage
def main():
    # Sample input
    answer = "The capital of France is Paris."

    # Initialize transformer
    transformer = ToneTransformer()
    
    # Transform to a specific tone
    transformer.print_transformed_answer(answer, "academic")
    
    # Optionally, transform to all tones
    transformer.print_all_transformations(answer)

if __name__ == "__main__":
    main()