import torch
import numpy as np
import scipy
import nltk
import typing
from difflib import SequenceMatcher
from evaluate import load
from .bleu.bleu import Bleu
import logging

MATCH_METRICS = ['exact_match', 'bleu', 'rougeL']


def format_ratio(pre_datum, post_datum):
    sign_prefix = ('+' if post_datum >= pre_datum else '')
    abs_ratio = sign_prefix + f'{format_score((post_datum - pre_datum) * 100)}%'
    rel_ratio = sign_prefix + f'{format_score((post_datum / pre_datum - 1.) * 100)}%'
    return abs_ratio, rel_ratio

def format_score(datum):
    return round(datum, 3)

class Metric:
    @staticmethod
    def exact_match(gens: list[list[str]], refs: list[list[str]]):
        """Exact Match on the token-level"""
        score = np.prod([1 if g == r else 0 for g, r in zip(gens, refs)])
        return format_score(float(score))

    # @staticmethod
    # def broad_match(gens: list[list[str]], refs: list[list[str]]):
    #     """Broad Match on the token-level"""
    #     score = np.sum([1 if g == r else 0 for g, r in zip(gens, refs)]) / len(refs)
    #     return format_score(score)

    # @staticmethod
    # def longest_match(gens: list[list[str]], refs: list[list[str]]):
    #     """Longest Common Substring on the token-level"""
    #     gen, ref = tuple(gens), tuple(refs)
    #     matcher = SequenceMatcher(a=gen, b=ref, autojunk=False)
    #     match = matcher.find_longest_match()
    #     score = match.size / len(ref)
    #     return format_score(score)

    @staticmethod
    def bleu_score(predictions: list[str], references: list[str]):
        """BLEU on the token-level"""
        predictions = [prediction for prediction in predictions]
        references = [[reference] for reference in references]
        metric = Bleu()
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['bleu']
        return format_score(score)

    @staticmethod
    def rouge_score(predictions: list[str], references: list[str]):
        """ROUGE on the token-level"""
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        metric = load('./codellmeditor/evaluate/rouge')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['rougeL']
        return format_score(score)

    @staticmethod
    def singular_scoring(actual_gens, oracle_gens):
        for func in (Metric.exact_match, Metric.broad_match):
            actual_score = func(actual_gens, oracle_gens)
            logging.success(f'{func.__name__}: {actual_score=}')


def batch_generate(model, tok, prompts, max_length, sample_generate = False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    gen_args = {
            'input_ids': prompt_tok['input_ids'],
            'attention_mask': prompt_tok['attention_mask'],
            'max_new_tokens': max_length,
            'pad_token_id': tok.eos_token_id,
        }
    with torch.no_grad():
        if sample_generate:
            gen_args.update({
                'do_sample': True,
                'num_beams': 1,
                'top_k': 5,
            })
        gen_tokens = model.generate(**gen_args)
    return tok.batch_decode(gen_tokens[:, prompt_tok['input_ids'].shape[1]:])



def test_generation_quality(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    tokenizer_for_fluency=None
):
    gen_texts = batch_generate(
        model,
        tok,
        prefixes,
        max_out_len,
        True
    )
    ngram_entropy = n_gram_entropy(gen_texts, tokenizer_for_fluency=tokenizer_for_fluency)
    ret = {
        "ngram_entropy": ngram_entropy,
        "generated_texts" : gen_texts
    }
    return ret


def n_gram_entropy(gen_texts, agg="arith", tokenizer_for_fluency=None):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt, tokenizer_for_fluency=tokenizer_for_fluency) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith", tokenizer_for_fluency=None):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n, tokenizer_for_fluency)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2, tokenizer_for_fluency=None):
    if tokenizer_for_fluency is not None:
        tokens = tokenizer_for_fluency.encode(sentence)
    else:
        tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)

