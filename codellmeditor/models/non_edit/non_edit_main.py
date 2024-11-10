from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .non_edit_hparams import NON_EDITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE= {}


def apply_non_edit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NON_EDITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    return model, {}
    

