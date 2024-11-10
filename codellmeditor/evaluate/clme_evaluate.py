"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
import re
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
import numpy as np
from .evaluate_utils import (
    batch_generate,
    Metric,
    test_generation_quality,
    MATCH_METRICS
)
import logging

LOG = logging.getLogger(__name__)
def compute_edit_quality(
    model,
    tok: AutoTokenizer,
    record: typing.Dict,
    test_generation: bool,
    tokenizer_for_fluency=None,
) -> typing.Dict:
    # First, unpack rewrite evaluation record.
    intent = record['prompt']
    rewritten_intent = record['rephrase_prompt']
    target_snippet = record['target_new']
    neighborhoods = record['specificity']
    ret = {}
    ret['gen_strs'] = []
    ## Test efficacy and generalization
    target_snippet_token = tok.encode(target_snippet)
    gen_strs = batch_generate(model, tok, [intent, rewritten_intent], max_length=len(target_snippet_token))
    ret['gen_strs'].append(gen_strs)
    ret['efficacy'] = {}
    ret['generalization'] = {}    
    for i, func in enumerate([
                Metric.exact_match,
                Metric.bleu_score,
                Metric.rouge_score,
        ]):
        if gen_strs[0].strip() == '':
            ret['efficacy'][MATCH_METRICS[i]] = 0
        else:
            ret['efficacy'][MATCH_METRICS[i]] = func([gen_strs[0].strip()], [target_snippet])
        if gen_strs[1].strip() == '':
            ret['generalization'][MATCH_METRICS[i]] = 0
        else:
            ret['generalization'][MATCH_METRICS[i]] = func([gen_strs[1].strip()], [target_snippet])
    
    ## Test specificity
    ret['specificity'] = {}   
    gen_strs = batch_generate(model, tok, neighborhoods['prompts'], max_length=16)
    for i, func in enumerate([
            Metric.exact_match,
            Metric.bleu_score,
            Metric.rouge_score,
    ]):
        score = func(gen_strs, neighborhoods['ground_truth'])
        ret['specificity'][MATCH_METRICS[i]] = np.mean(score)
    if test_generation:
        try:
            res = test_generation_quality(model, tok, [intent, rewritten_intent], max_out_len=64)
            ret.update(res)
        except Exception as e:
            LOG.warn(f"Case {record['case_id']} test generation raise error {e}")
    return ret

