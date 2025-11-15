import re
import string
from collections import Counter

import nltk
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer


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

        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

    def evaluate(self, reference: str, prediction: str) -> dict:
        # ROUGE Scores
        rouge_scores = self.rouge_scorer.score(reference, prediction)

        # BLEU Score
        bleu = sentence_bleu(
            [reference.split()], prediction.split(), smoothing_function=self.smoothing_function
        )

        # METEOR Score
        reference_tokens = nltk.word_tokenize(reference)
        prediction_tokens = nltk.word_tokenize(prediction)
        meteor = single_meteor_score(reference_tokens, prediction_tokens)

        # BERTScore
        P, R, F1 = score([prediction], [reference], lang="en", verbose=False)
        bert_f1 = F1.mean().item()

        # Token F1 Score
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
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    golden_text = '''unanswerable: False\nextractive_spans: None\nyes_no: None\nfree_form_answer: 30 words\nevidence:\n  - We constructed our seed lexicon consisting of 15 positive words and 15 negative words, as shown in Section SECREF27. From the corpus of about 100 million sentences, we obtained 1.4 millions event pairs for AL, 41 millions for CA, and 6 millions for CO. We randomly selected subsets of AL event pairs such that positive and negative latter events were equal in size. We also sampled event pairs for each of CA and CO such that it was five times larger than AL. The results are shown in Table TABREF16.\nhighlighted_evidence:\n  - We constructed our seed lexicon consisting of 15 positive words and 15 negative words, as shown in Section SECREF27. '''

    predicted_text = '''unanswerable: False\nextractive_spans:\n  - 15 positive words and 15 negative words\nyes_no: None\nfree_form_answer:\n\nevidence:\n  - We constructed our seed lexicon consisting of 15 positive words and 15 negative words, as shown in Section A.1.\nhighlighted_evidence:\n  - We constructed our seed lexicon consisting of 15 positive words and 15 negative words, as shown in Section A.1. '''

    evaluator = MetricsEvaluator()
    results = evaluator.evaluate(golden_text, predicted_text)
    evaluator.display_results(results)
