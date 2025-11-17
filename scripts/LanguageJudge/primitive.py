import os
import openai
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMJudge:
    def __init__(self):
        """Initialize the LLM Judge with OpenAI API configuration"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"

    def create_evaluation_prompt(self, context: str, question: str, 
                              answer: str, instructions: Optional[str] = None) -> str:
        """Create a detailed prompt for evaluating the QNA response"""
        prompt = """
You are an expert evaluator tasked with assessing the quality of a question-answering system's response. Your job is to provide scores from 1-10 (where 1 is poor and 10 is excellent) for four specific criteria: Correctness, Fluency/Coherence, Groundedness, and Format. Please analyze the provided context, question, answer, and any additional instructions carefully. Return your evaluation in a structured JSON format.

Here are the criteria definitions:
1. Correctness: How accurate and factually correct is the answer in relation to the question and context?
2. Fluency/Coherence: How well-written, clear, and logically coherent is the answer?
3. Groundedness: How well does the answer stay rooted in the provided context and avoid hallucination or irrelevant information?
4. Format: How well-structured and appropriately presented is the answer?

Input Information:
- Context: {context}
- Question: {question}
- Answer: {answer}
{instructions_section}

Evaluation Instructions:
1. Analyze the answer in relation to the context and question.
2. Consider any provided instructions that the answering model was supposed to follow.
3. Provide a score (1-10) for each criterion.
4. Include a brief explanation for each score.
5. Return the results in this exact JSON format:
{
    "correctness": {
        "score": <int>,
        "explanation": "<string>"
    },
    "fluency_coherence": {
        "score": <int>,
        "explanation": "<string>"
    },
    "groundedness": {
        "score": <int>,
        "explanation": "<string>"
    },
    "format": {
        "score": <int>,
        "explanation": "<string>"
    }
}

Please provide your evaluation based on the given information.
""".format(
    context=context,
    question=question,
    answer=answer,
    instructions_section=f"- Instructions for answering model: {instructions}" if instructions else ""
)
        return prompt

    def evaluate_response(self, context: str, question: str, 
                         answer: str, instructions: Optional[str] = None) -> Dict:
        """Evaluate the QNA response using GPT-4o-mini"""
        try:
            prompt = self.create_evaluation_prompt(context, question, answer, instructions)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise and analytical evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent, analytical responses
                max_tokens=1000
            )
            
            # Extract and parse the JSON response
            evaluation_text = response.choices[0].message.content
            # Assuming the response is properly formatted JSON
            import json
            evaluation_result = json.loads(evaluation_text)
            
            return evaluation_result

        except Exception as e:
            return {
                "error": f"Evaluation failed: {str(e)}",
                "correctness": {"score": 0, "explanation": "Evaluation error"},
                "fluency_coherence": {"score": 0, "explanation": "Evaluation error"},
                "groundedness": {"score": 0, "explanation": "Evaluation error"},
                "format": {"score": 0, "explanation": "Evaluation error"}
            }

    def print_evaluation(self, evaluation: Dict):
        """Print the evaluation results in a formatted way"""
        if "error" in evaluation:
            print(f"Error: {evaluation['error']}")
            return
        
        print("\nEvaluation Results:")
        for criterion, details in evaluation.items():
            print(f"{criterion.replace('_', ' ').title()}:")
            print(f"  Score: {details['score']}/10")
            print(f"  Explanation: {details['explanation']}\n")


# Example usage
def main():
    # Sample input
    context = "The capital city of France is Paris, which is not only the largest city in France but also a major cultural center."
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    instructions = "Answer concisely in one sentence."

    # Initialize judge
    judge = LLMJudge()
    
    # Evaluate response
    evaluation = judge.evaluate_response(context, question, answer, instructions)
    
    # Print results
    judge.print_evaluation(evaluation)

if __name__ == "__main__":
    main()