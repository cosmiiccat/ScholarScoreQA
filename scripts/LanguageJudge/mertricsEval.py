import re
import string
import json
import boto3
import nltk
import requests
from collections import Counter
from typing import List
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class TextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return " ".join(text.split())

class TokenF1Score:
    @staticmethod
    def compute(prediction: str, ground_truth: str) -> float:
        pred_tokens = TextNormalizer.normalize(prediction).split()
        gt_tokens = TextNormalizer.normalize(ground_truth).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

class MetricsEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method2

    def evaluate(self, reference: str, prediction: str) -> dict:
        rouge_scores = self.rouge_scorer.score(reference, prediction)
        bleu = sentence_bleu(
            [reference.split()], prediction.split(), smoothing_function=self.smoothing_function
        )
        reference_tokens = nltk.word_tokenize(reference)
        prediction_tokens = nltk.word_tokenize(prediction)
        meteor = single_meteor_score(reference_tokens, prediction_tokens)
        P, R, F1 = score([prediction], [reference], lang="en", verbose=False)
        bert_f1 = F1.mean().item()
        span_f1 = TokenF1Score.compute(prediction, reference)
        return {
            "ROUGE-1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-2": rouge_scores['rouge2'].fmeasure,
            "ROUGE-L": rouge_scores['rougeL'].fmeasure,
            "BLEU": bleu,
            "METEOR": meteor,
            "BERTScore": bert_f1,
            "Token F1": span_f1,
        }

    def display_results(self, metrics: dict):
        print("Best Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

class S3GraphQLFetcher:
    def __init__(self, graphql_endpoint: str, api_key: str):
        self.graphql_endpoint = graphql_endpoint
        self.api_key = api_key

    def fetch_golden_texts(self, question: str) -> List[str]:
        query = """
        query ListGoldens($question: String!) {
          listGoldens(filter: {question: {eq: $question}}) {
            items {
              context
              question
              answer
            }
          }
        }
        """
        payload = {
            "query": query,
            "variables": {
                "question": question
            }
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        response = requests.post(self.graphql_endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        items = data.get("data", {}).get("listGoldens", {}).get("items", [])
        golden_texts = [item["context"] for item in items if "context" in item]
        return golden_texts

def main():
    import os
    graphql_endpoint = os.environ["GRAPHQL_ENDPOINT"]
    api_key = os.environ["GRAPHQL_API_KEY"]
    bucket_name = os.environ["BUCKET_NAME"]
    prefix = os.environ["PREFIX"]
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

    obj = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    contents = obj.get('Contents', [])

    key = contents[0]["Key"]
    predicted_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    predicted_text = predicted_obj['Body'].read().decode('utf-8')

    question = os.environ["QUESTION"]

    fetcher = S3GraphQLFetcher(graphql_endpoint, api_key)
    golden_texts = fetcher.fetch_golden_texts(question)

    evaluator = MetricsEvaluator()
    best_metrics = None

    for golden in golden_texts:
        metrics = evaluator.evaluate(golden, predicted_text)
        if not best_metrics:
            best_metrics = metrics
        for metric_name, _ in metrics.items():
            if metrics[metric_name] > best_metrics[metric_name]:
                best_metrics[metric_name] = metrics[metric_name]

    if best_metrics:
        evaluator.display_results(best_metrics)

if __name__ == "__main__":
    main()
