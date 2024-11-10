import logging
import re

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import scr

LOG = logging.getLogger(__name__)


class CastModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        in_cast: torch.dtype = torch.float32,
        out_cast: torch.dtype = None,
    ):
        super().__init__()

        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f"Not sure how to cast type {type(outputs)}")
        return outputs

    def extra_repr(self):
        return f"in_cast: {self.in_cast}\nout_cast: {self.out_cast}"


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(model_name, cache_dir=scr())
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        return self.classifier(self.model(*args, **filtered_kwargs)[1])


def get_model(config):
    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True, device_map='auto' if config.model_parallel else None)

    if config.half:
        model.bfloat16()
    return model


def get_tokenizer(config):
    tok_name = (
        config.tokenizer_name
        if config.tokenizer_name is not None
        else config.model.name
    )
    tokenizer =  AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token_id  = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    return tokenizer


if __name__ == "__main__":
    m = BertClassifier("bert-base-uncased")
    m(torch.arange(5)[None, :])
    import pdb

    pdb.set_trace()
