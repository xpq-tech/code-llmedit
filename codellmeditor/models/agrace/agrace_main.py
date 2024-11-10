from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .AGRACE import AGRACE
from .agrace_hparams import AGraceHyperParams
from .utils import tokenize
from ...util import nethook


def apply_agrace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: AGraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)
    weights_copy = {}
    device = torch.device(f'cuda:{hparams.device}')
    editor = AGRACE(model=model, config=hparams, device=device)
    tokens = tokenize(request, tokenizer=tok, device=device)
    if "CodeQwen1.5-7B" in model.config.name_or_path:
        tokens.pop('token_type_ids')
    editor.edit(config=hparams, tokens=tokens,edit_id=request['target_new'])
    # editor.rolllback(request['target_new'])


    with torch.no_grad():
        for w_name in hparams.inner_params:
            w_name=w_name.replace("[", ".").replace("]", "")
            w = nethook.get_parameter(editor.model, w_name)
            weights_copy[w_name]=w
            
    if keep_original_weight:
        weights_copy = editor.reset_layer


    return editor, weights_copy


