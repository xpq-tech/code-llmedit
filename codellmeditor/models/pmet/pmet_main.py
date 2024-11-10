import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *
from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import compute_zs, get_module_input_output_at_words, find_fact_lookup_idx
from .pmet_hparams import PMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE= {}

LOG = logging.getLogger(__name__)


def apply_pmet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    
    deltas = execute_pmet(model, tok, requests, hparams, cache_template=cache_template) 

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items(): #w_name, adj_k, resid
            upd_matrix = upd_matrix.to("cuda")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float() #w[...]高级索引，表示对w中每个元素进行操作

    LOG.debug(f"\nNew weights successfully inserted into {list(deltas.keys())}")
    if not keep_original_weight:
        weights_copy = {}
    return model, weights_copy


def execute_pmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        last_word = request['prompt'].split(' ')[-1]
        request['subject'] = last_word
        request['prompt'] = ' '.join(request['prompt'].split(' ')[:-1]) + f" {PADD_STR}"
    for request in requests[:10]:
        LOG.debug(
            f"PMET request sample: "
            f"[{request['prompt'].replace(PADD_STR, request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( # transformer.h.{}.attn.out_proj
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = dict()
    for rewrite_module_name in rewrite_module_names:
        z_list[rewrite_module_name] = []
    # get zs
    for request in requests:
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    z_list[rewrite_module_name].append(torch.from_numpy(data["v_star"]).to("cuda"))
                    data_loaded = True
                except Exception as e:
                    LOG.info(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                if len(rewrite_module_names) == 2:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates,
                    )
                    z_list[rewrite_module_names[0]].append(cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp)
                    z_list[rewrite_module_names[1]].append(cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp)
                    for rewrite_module_name in rewrite_module_names:
                        block_name = "attn" if "attn" in rewrite_module_name else "mlp"
                        cache_fname = (
                            Path(
                                str(cache_template).format(
                                    z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                                )
                            )
                            if cache_template is not None
                            else None
                        )
                        if cache_fname is not None:
                            cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            if block_name == "attn":
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_attn.detach().cpu().numpy(),
                                    },
                                )
                            else:
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_mlp.detach().cpu().numpy(),
                                    },
                                )
                            LOG.debug(f"Cached k/v pair at {cache_fname}")
                else:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                    if "attn" == block_name:
                        cur_z = cur_z_attn
                    else:
                        cur_z = cur_z_mlp
                    z_list[rewrite_module_name].append(cur_z)
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(
                            cache_fname,
                            **{
                                "v_star": cur_z.detach().cpu().numpy(),
                            },
                        )
                        LOG.debug(f"Cached k/v pair at {cache_fname}")
                break

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        LOG.debug(f"\n\nLAYER {layer}\n") 
        layers_ks = None
        # force_recompute = layer != hparams.layers[0]
        for rewrite_module_name in rewrite_module_names:
            # Get current model activations
            layers_ks = compute_ks(model, tok, requests, hparams, rewrite_module_name, layer, context_templates)

            LOG.debug(f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers")
            cur_zs = get_module_input_output_at_words( # hidden states eqn 2
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = z_list[rewrite_module_name]  - cur_zs #z_i - h_i^L
            try:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double(),
                    targets.double()
                )
            except:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double().to(f'cuda:{hparams.device}'),
                    targets.double().to(f'cuda:{hparams.device}')
                )
            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
                hparams=hparams
            )

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1) #r
            upd_matrix =  (targets / np.sqrt((len(hparams.layers) - i ))) @ layer_ks.T @ torch.inverse(layer_ks @ layer_ks.T + 
                                             hparams.mom2_update_weight * cov.double())
            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            LOG.debug(f"{weight_name} :\norig norm {torch.linalg.norm(weights[weight_name])}")
            LOG.debug(f"upd norm {torch.linalg.norm(upd_matrix)}")

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to(f'cuda:{hparams.device}')
                deltas[weight_name] = upd_matrix

            # Clear GPU memory

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    LOG.debug(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    LOG.debug(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats( # download
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams,
            batch_tokens=4096
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    try:
        return (
            torch.inverse(COV_CACHE[key].to("cuda:0")) if inv else COV_CACHE[key].to("cuda:0")
        )
    except:
        return (
            torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
        )
def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [[f"{PADD_STR}"]] + [
            [
                f + f". {PADD_STR}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                ) # 用模型生成句子
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        LOG.debug(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
