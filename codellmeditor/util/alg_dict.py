from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.agrace import AGraceHyperParams, apply_agrace_to_model
from ..models.pmet import PMETHyperParams, apply_pmet_to_model
from ..models.non_edit import NON_EDITHyperParams, apply_non_edit_to_model
from ..models.malmen import MALMENHyperParams, MalmenRewriteExecutor
from ..models.memit import MEMITHyperParams, apply_memit_to_model


ALG_DICT = {
    'ROME': apply_rome_to_model,
    "FT": apply_ft_to_model,
    'GRACE': apply_grace_to_model,
    'AGRACE': apply_agrace_to_model,
    'PMET': apply_pmet_to_model,
    'non-edit': apply_non_edit_to_model,
    'MALMEN': MalmenRewriteExecutor().apply_to_model,
    "MEMIT": apply_memit_to_model
}
