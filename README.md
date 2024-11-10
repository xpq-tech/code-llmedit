# CLMEEval

This is a repository of our paper "Model Editing for LLMs4Code: How Far are We?"

## Requirements

#### Pip Installation

**Note: Please use Python 3.9+ for CLMEEval.**
To get started, simply install conda and run:

```shell
conda create -n clme python=3.9.7
...
pip install -r requirements.txt
```
## Data Download

The **CNLE** and **CSNE** datasets are available on [Zenodo](https://doi.org/10.5281/zenodo.12818365).

* Note: If you are using editing techniques that require training (e.g., MALMEN and A-GRACE), please divide the dataset into training and testing sets first.

## Run Experiments

### Editing CodeLlama on the CNLE dataset using A-GRACE
First, use [train_agrace_encoder.sh](./train_agrace_encoder.sh) to train the encoder for A-GRACE. Then, use the following script to edit CodeLlama on the CNLE dataset using A-GRACE.


```shell
python edit_main.py \
    --editing_method=AGRACE \
    --data_dir=./[your splitted data dir] \
    --data_set=EditConala \
    --hparams_dir=./hparams/AGRACE/codellama-7b.yaml
```

### Run other experiments
Use the following script template to run experiments:

```shell
python edit_main.py \
--editing_method=[Editing Approach] \
--hparams_dir=[Hparams Path]
```

All run scripts are available in [run_scripts](./run_scripts.sh).


**Note:** Ensure that the paths are set appropriately on your device.

## Acknowledgement
This project is derived from [EasyEdit](https://github.com/zjunlp/EasyEdit)
