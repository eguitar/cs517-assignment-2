import re
import string
from collections import Counter

def normalize_text(s):
    """Lowercases, removes punctuation and articles per SQuAD section 6.1."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_recall(prediction, truth):
    """Proportion of reference tokens that appear in the prediction."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(truth_tokens) == 0:
        return int(len(pred_tokens) == 0)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    return num_same / len(truth_tokens)

def get_scores(prediction, reference_line):
    """
    Handles semicolon-separated multiple reference answers.
    Takes the max score across all valid answers per SQuAD section 6.1.
    """
    references = [r.strip() for r in reference_line.split(';')]
    em  = max(compute_exact_match(prediction, ref) for ref in references)
    f1  = max(compute_f1(prediction, ref)          for ref in references)
    rec = max(compute_recall(prediction, ref)       for ref in references)
    return em, f1, rec

def evaluate(system_output_path, reference_answers_path):
    """
    Reads both files line by line and returns macro-averaged EM, F1, and Recall.
    """
    with open(system_output_path, encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    with open(reference_answers_path, encoding='utf-8') as f:
        references = [line.strip() for line in f]

    assert len(predictions) == len(references), (
        f"Line count mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    total_em, total_f1, total_rec = 0, 0, 0
    for pred, ref in zip(predictions, references):
        em, f1, rec = get_scores(pred, ref)
        total_em  += em
        total_f1  += f1
        total_rec += rec

    n = len(predictions)
    results = {
        "exact_match": round(total_em  / n * 100, 2),
        "f1":          round(total_f1  / n * 100, 2),
        "recall":      round(total_rec / n * 100, 2),
        "num_questions": n
    }
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py reference_answers.txt system_output.txt")
        sys.exit(1)
    results = evaluate(sys.argv[2], sys.argv[1])

    descriptions = {
        "exact_match": "predictions that matched a reference answer word-for-word",
        "f1":          "average token overlap between predictions and reference answers (main metric)",
        "recall":      "reference answer tokens successfully captured in predictions",
        "num_questions": "total questions evaluated"
    }

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        if k == "num_questions":
            value_str = str(v)
        else:
            value_str = f"{v}%"
        print(f"  {k:<20} {value_str:<10} {descriptions[k]}")
    print("--------------------------\n")