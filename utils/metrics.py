import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import defaultdict
from bert_score import score as bert_score
import torch
from models.constants import BERT_1, BERT_2
nltk.download('punkt')

def calculate_all_scores(reference_answers, generated_answers):
    # Convert answers to string
    reference_answers = [str(answer) for answer in reference_answers]
    generated_answers = [str(answer) for answer in generated_answers]

    # Calculate BLEU Score
    def calculate_bleu(reference_answers, generated_answers):
        smoothing_function = SmoothingFunction().method1
        return np.mean([sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(gen), smoothing_function=smoothing_function) for ref, gen in zip(reference_answers, generated_answers)])

    bleu_score = calculate_bleu(reference_answers, generated_answers)

    # Calculate Average ROUGE Scores
    def calculate_average_rouge(reference_answers, generated_answers):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, gen) for ref, gen in zip(reference_answers, generated_answers)]
        
        total_scores = defaultdict(lambda: defaultdict(float))
        num_scores = len(scores)
        
        for score in scores:
            for rouge_type in score:
                total_scores[rouge_type]['precision'] += (score[rouge_type].precision)*100
                total_scores[rouge_type]['recall'] += (score[rouge_type].recall)*100
                total_scores[rouge_type]['fmeasure'] += (score[rouge_type].fmeasure)*100
        
        return {rouge_type: {k: total_scores[rouge_type][k] / num_scores for k in total_scores[rouge_type]} for rouge_type in total_scores}

    average_rouge_scores = calculate_average_rouge(reference_answers, generated_answers)

    # Calculate BERTScore
    def calculate_bertscore(reference_answers, generated_answers, model_type):
        truncated_references = [ref[:512] for ref in reference_answers]
        truncated_hypotheses = [hyp[:512] for hyp in generated_answers]
        P, R, F1 = bert_score(cands=truncated_hypotheses, refs=truncated_references, model_type=model_type, num_layers=12, idf=False, batch_size=3, device='cuda' if torch.cuda.is_available() else 'cpu')
        return P.mean().item()*100, R.mean().item()*100, F1.mean().item()*100

    bert_scores_biobert = calculate_bertscore(reference_answers, generated_answers, BERT_1)
    bert_scores_pubmedbert = calculate_bertscore(reference_answers, generated_answers, BERT_2)

    return {
        "BLEU": bleu_score * 100,
        "ROUGE": average_rouge_scores,
        "BERTScore_BioBERT": bert_scores_biobert,
        "BERTScore_PubMedBERT": bert_scores_pubmedbert
    }
