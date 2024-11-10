from codellmeditor import (
    FTHyperParams, 
    PMETHyperParams,
    MALMENHyperParams,
    GraceHyperParams,
    NON_EDITHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    AGraceHyperParams
    )
from codellmeditor import BaseEditor, CLMEDataset, EditTrainer, prepare_requests
import argparse
import logging
import gc
import torch


LOG = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default='MALMEN', type=str)
    parser.add_argument('--hparams_dir', default='./hparams/MALMEN/stable-code-3b.yaml', type=str)
    parser.add_argument('--data_dir', default='./data/data_loo1', type=str)
    parser.add_argument('--data_set', default='EditConala', type=str, choices=["EditConala", "EditCodeSearchNet"])
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--generation_test_interval', default=0, type=int)
    parser.add_argument('--continue_from_run', default=None, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--log_level', default='INFO', type=str)

    args = parser.parse_args()
    log_level = logging.INFO
    if args.log_level == 'DEBUG':
        log_level = logging.DEBUG
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = log_level)
    train_datas = None
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'AGRACE':
        editing_hparams = AGraceHyperParams
    elif args.editing_method == 'PMET':
        editing_hparams = PMETHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MALMEN':
        editing_hparams = MALMENHyperParams
        train_datas = CLMEDataset(f"{args.data_dir}/{args.data_set}/train.json")
    elif args.editing_method == 'non-edit':
        editing_hparams = NON_EDITHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)


    if args.editing_method in ['MALMEN', 'AGRACE']:
        train_datas = CLMEDataset(f"{args.data_dir}/{args.data_set}/train.json", tokenizer_name=hparams.tokenizer_name)
        test_datas = CLMEDataset(f"{args.data_dir}/{args.data_set}/test.json", size=args.ds_size)
    else:
        test_datas = CLMEDataset(f"{args.data_dir}/{args.data_set}/all.json", size=args.ds_size)
    

    test_datas = prepare_requests(test_datas, hparams.model_name, args.data_set)
    if args.editing_method == 'MALMEN' and hparams.archive is None:
        train_datas = prepare_requests(train_datas, hparams.model_name, args.data_set)
        trainer = EditTrainer(
            config=hparams,
            train_set=train_datas,
            val_set=test_datas
        )
        trainer.run()
        hparams.archive = trainer.save_path
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    editor = BaseEditor.from_hparams(hparams, args.data_set)
    metrics, edited_model, _ = editor.edit(
        requests = test_datas,
        keep_original_weight = True,
        generation_test_interval = args.generation_test_interval,
        data_set_name = args.data_set,
        continue_from_run = args.continue_from_run
    )
