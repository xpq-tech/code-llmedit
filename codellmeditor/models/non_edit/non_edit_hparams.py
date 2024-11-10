from dataclasses import dataclass
from typing import List, Literal

from ...util.hparams import HyperParams
import yaml


@dataclass
class NON_EDITHyperParams(HyperParams):
    alg_name: str
    model_name: str
    device: int
    fp16: bool = False
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'non-edit') or print(f'NON_EDITHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
