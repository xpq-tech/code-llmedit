import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from datetime import datetime
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
# from accelerate import Accelerator
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, MATCH_METRICS
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

LOG = logging.getLogger(__name__)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def make_logs(log_name = None):
    if log_name is None:
        log_name = 'run.log'
    f_h, s_h = get_handler('logs', log_name)
    for h in LOG.handlers:
        LOG.removeHandler(h)
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams,  data_set_name):

        return cls(hparams, data_set_name)

    def __init__(self,
                hparams: HyperParams,
                data_set_name,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        time_now = datetime.now().strftime("%m-%d-%H-%M")
        make_logs(f"{self.alg_name}_{self.model_name.split('/')[-1]}_{data_set_name}_{time_now}.log")

        LOG.debug("Instantiating model")
        device_map = 'auto' if hparams.model_parallel else None
        torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
        self.tok = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tok.add_bos_token = False
        self.tok.pad_token_id = self.tok.eos_token_id
        self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             requests,
             data_set_name,
             generation_test_interval: Optional[int] = 0,
             keep_original_weight=False,
             continue_from_run = None
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        edited_model, weights_copy = None, None
        results_dir = Path(f"./results/{data_set_name}/{self.alg_name}/{self.model_name.split('/')[-1]}")
        if continue_from_run:
            run_id = continue_from_run
            run_file = results_dir / f"{continue_from_run}.json"
            if run_file.exists():
                all_metrics = json.load(open(run_file))
                computed_cases = [item['case_id'] for item in all_metrics]
                output_file = run_file
        else:
            if results_dir.exists():
                id_list = [
                    int(str(x).split("_")[-1][:3])
                    for x in results_dir.iterdir()
                    if str(x).split("_")[-1][:3].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
                os.makedirs(results_dir)  
            all_metrics = []    
            output_file = results_dir / f"run_{str(run_id).zfill(3)}.json"
        LOG.info(f"Results will be stored in {output_file}")
        if data_set_name == "EditConala":
            codebert_tokenizer = AutoTokenizer.from_pretrained("../ptms/codebert-base-mlm")
        else:
            codebert_tokenizer = None
        for index, request in tqdm(enumerate(requests), total=len(requests)):
            if continue_from_run:
                if request["case_id"] in computed_cases:
                    LOG.debug(f"Case {request['case_id']} already exists.")
                    continue
            start = time()
            try:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                )
            except Exception as e:
                with open(output_file, 'w') as f:  
                    json.dump(all_metrics, f, ensure_ascii=False, indent=4)
                raise RuntimeError(e)
            exec_time = time() - start
            LOG.info(f"Execution {request['case_id']} editing took {exec_time}")

            all_metrics.append({
                'case_id': request['case_id'],
                # "requested_rewrite": request,
                "time": exec_time,
                "max_memory": torch.cuda.max_memory_allocated(f"cuda:{self.hparams.device}")/ 1024**2,
                "post": compute_edit_quality(edited_model, self.tok, request, test_generation = (generation_test_interval % (1+request['case_id']) == 0), tokenizer_for_fluency=codebert_tokenizer),
            })
            torch.cuda.reset_peak_memory_stats(f"cuda:{self.hparams.device}")
            if 'GRACE' in self.alg_name and keep_original_weight:
                with torch.no_grad():
                    weights_copy() # unpatch_fn
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            LOG.debug(
                    f"{request['case_id']} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[-1]} \n max memory: {torch.cuda.max_memory_allocated(f'cuda:{self.hparams.device}')/ 1024**2:.2f}M"
                )
            if (index+1)%50 == 49:
                with open(output_file, 'w') as f:  
                    json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
        with open(output_file, 'w') as f:  
            json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
        mean_metrics = dict()
        mean_metrics['run_id'] = run_id
        for metric in ['efficacy', 'generalization', 'specificity']:
            mean_metrics[metric] = dict()
            for match_metric in MATCH_METRICS:
                mean_metrics[metric][match_metric] = (np.round(np.mean([item['post'][metric][match_metric] for item in all_metrics])*100, 2), np.round(np.std([item['post'][metric][match_metric] for item in all_metrics])*100, 2))
        ngram_entropys = []
        for item in all_metrics:
            if 'ngram_entropy' in item['post']:
                ngram_entropys.append(item['post']['ngram_entropy'])
        mean_metrics['fluency'] = (np.round(np.mean(ngram_entropys)*100, 2), np.round(np.std(ngram_entropys)*100, 2))
        mean_metrics["time"] = (np.round(np.mean([metric["time"] for metric in all_metrics]),3), np.round(np.std([metric["time"] for metric in all_metrics]),3))
        mean_metrics["max_memory"] = (np.round(np.mean([metric["max_memory"] for metric in all_metrics]),3), np.round(np.std([metric["max_memory"] for metric in all_metrics]),3))
        mean_metrics["hparams"] = str(self.hparams)
        mean_metrics_save_dir = results_dir / f"mean_run_{str(run_id).zfill(3)}.json"
        with open(mean_metrics_save_dir, 'w') as f:  
            json.dump(mean_metrics, f, ensure_ascii=False)
        LOG.info(f"Run {run_id}\nMetrics Summary: {mean_metrics}")
        LOG.info(self.hparams)

        return all_metrics, edited_model, weights_copy

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_prompts: Optional[List[str]] = None,
                   locality_ground_truth: Optional[List[str]] = None,
                   keep_original_weight=False,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False


        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, rephrase_prompts,
                                          locality_prompts, locality_ground_truth, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')
        all_metrics = []
        for record_chunks in self._chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }

                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)
                LOG.debug(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in DS_DICT.values()]) > 0, print(f'DataSet {ds} not supported yet.')

        is_singleton = SingletonEditor.is_singleton_method(self.alg_name)

        if is_singleton:
            num_edits = 1 # Single editor method found
        else:
            assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls set the batch_size correctly')

            num_edits = self.hparams.batch_size

        all_metrics = []

        for record_chunks in tqdm(self._chunks(ds, num_edits), desc='Editing dataset', total=len(ds)/num_edits):
            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight
            )
            exec_time = time() - start
            LOG.debug(f"Execution took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': request['case_id'],
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }
                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                      self.hparams.device)

                LOG.debug(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.debug(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]


    def edit_requests(self,
             requests,
             keep_original_weight=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.debug(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                LOG.debug(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        for i, request in enumerate(tqdm(requests)):
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                metrics = {
                    "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
            else:
                metrics = {
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                }
            all_metrics.append(metrics)

        for i, request in enumerate(tqdm(requests)):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.debug(f"Execution {i} editing took {exec_time}")
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
                all_metrics[i]['pre'].pop('locality')

                LOG.debug(f"Evaluation took {time() - start}")
                LOG.debug(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.debug(f"Execution {i} editing took {exec_time}")

                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if self.alg_name == 'KN' or self.alg_name == 'GRACE':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                elif self.alg_name == 'LoRA' and keep_original_weight:
                    edited_model.unload()
                    del self.model.peft_config
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                if 'locality' in all_metrics[i]['post'].keys():
                    for locality_key in request['locality'].keys():
                        assert len(all_metrics[i]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[i]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans,label in zip(all_metrics[i]['post']['locality'][f'{locality_key}_output'],all_metrics[i]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[i]['pre'].pop('locality')

                LOG.debug(f"Evaluation took {time() - start}")
                LOG.debug(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        keep_original_weight=False,
        epoch: int=5,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=keep_original_weight,
        )
        exec_time = time() - start
        LOG.debug(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy

