#### Conala
### Non edit
# CUDA_VISIBLE_DEVICES=3 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/codellama-7b.yaml > logs/non_edit_codellama.log &

# CUDA_VISIBLE_DEVICES=2 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/CodeQwen1.5-7B.yaml  > logs/non_edit_codeqwen.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/stable-code-3b.yaml > logs/non_edit_stable-code-3b.log &

### FT-L
# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/codellama-7b.yaml > logs/ft_codellama-7b.log &

# CUDA_VISIBLE_DEVICES=2 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/CodeQwen1.5-7B.yaml > logs/ft_codeqwen1.5-7b.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/stable-code-3b.yaml > logs/ft_stable-code-3b.log &



### GRACE
# CUDA_VISIBLE_DEVICES=0 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/codellama-7b.yaml > logs/grace_codellama.log &

# CUDA_VISIBLE_DEVICES=0 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/CodeQwen1.5-7B.yaml > logs/grace_codeqwen1.5-7b.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/stable-code-3b.yaml  > logs/grace_stable-code-3b.log &

### PMET
# CUDA_VISIBLE_DEVICES=2 nohup python edit_main.py \
#     --editing_method=PMET \
#     --hparams_dir=./hparams/PMET/codellama-7b.yaml > logs/PMET_codellama.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=PMET \
#     --hparams_dir=./hparams/PMET/CodeQwen1.5-7B.yaml > logs/PMET_codeqwen.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=PMET \
#     --hparams_dir=./hparams/PMET/stable-code-3b.yaml > logs/PMET_stablecode.log &


### MALMEN
# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=MALMEN \
#     --data_dir=./[your splitted data dir]/ \
#     --hparams_dir=./hparams/MALMEN/codellama-7b.yaml > logs/malmen_codellama.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=MALMEN \
#     --data_dir=./[your splitted data dir] \
#     --hparams_dir=./hparams/MALMEN/CodeQwen1.5-7B.yaml > logs/malmen_CodeQwen.log &

# CUDA_VISIBLE_DEVICES=4 nohup python edit_main.py \
#     --editing_method=MALMEN \
#     --data_dir=./[your splitted data dir] \
#     --hparams_dir=./hparams/MALMEN/stable-code-3b.yaml > logs/malmen_stablecode.log &


### ROME
# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=ROME \
#     --hparams_dir=./hparams/ROME/codellama-7b.yaml> logs/rome_codellama.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=ROME \
#     --hparams_dir=./hparams/ROME/CodeQwen1.5-7B.yaml > logs/rome_CodeQwen.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=ROME \
#     --hparams_dir=./hparams/ROME/stable-code-3b.yaml > logs/rome_stablecode.log &

### MEMIT
# CUDA_VISIBLE_DEVICES=2 nohup python edit_main.py \
#     --editing_method=MEMIT \
#     --hparams_dir=./hparams/MEMIT/codellama-7b.yaml  > logs/memit_codellama-7b.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=MEMIT \
#     --hparams_dir=./hparams/MEMIT/CodeQwen1.5-7B.yaml > logs/memit_CodeQwen.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=MEMIT \
#     --hparams_dir=./hparams/MEMIT/stable-code-3b.yaml  > logs/memit_stable-code-3b.log &

### AGRACE
# CUDA_VISIBLE_DEVICES=3 nohup python edit_main.py \
#     --editing_method=AGRACE \
#     --data_dir=./[your splitted data dir] \
#     --hparams_dir=./hparams/AGRACE/codellama-7b.yaml > logs/agrace_codellama.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=AGRACE \
#     --data_dir=./[your splitted data dir] \
#     --hparams_dir=./hparams/AGRACE/CodeQwen1.5-7B.yaml > logs/agrace_codeqwen.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=AGRACE \
#     --data_dir=./[your splitted data dir] \
#     --hparams_dir=./hparams/AGRACE/stable-code-3b.yaml > logs/agrace_stable-code-3b.log &

# ================================================================================================================================================
### CodeSearchNet
# CUDA_VISIBLE_DEVICES=0 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/codellama-7b.yaml \
#     --data_set=EditCodeSearchNet  > logs/non_edit_codesearchnet_codellama.log &
    

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/non_edit_codesearchnet_codeqwen.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet > logs/non_edit_codesearchnet_stablecode.log &



### FT-L
# CUDA_VISIBLE_DEVICES=1 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/codellama-7b.yaml \
#     --data_set=EditCodeSearchNet > logs/ft_cs_codellama-7b.log &

# CUDA_VISIBLE_DEVICES=4 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/ft_cs_codeqwen1.5-7b.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet > logs/ft_cs_codesearch_stable-code-3b.log &


### GRACE
# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/codellama-7b.yaml \
#     --data_set=EditCodeSearchNet > logs/grace_cs_codellama.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/grace_cs_codeqwen1.5-7b.log &

# CUDA_VISIBLE_DEVICES=3 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet  > logs/grace_cs_stable-code-3b.log &

### PMET
# CUDA_VISIBLE_DEVICES=4 nohup python edit_main.py \
#     --editing_method=PMET \
#     --hparams_dir=./hparams/PMET/codellama-7b.yaml \
#     --data_set=EditCodeSearchNet > logs/PMET_cs_codellama.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=PMET \
#     --hparams_dir=./hparams/PMET/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/PMET_cs_codeqwen.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=PMET \
#     --hparams_dir=./hparams/PMET/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet > logs/PMET_cs_stablecode.log &

### MALMEN
# CUDA_VISIBLE_DEVICES=4 nohup python edit_main.py \
#     --editing_method=MALMEN \
#     --data_dir ./[your splitted data dir] \
#     --hparams_dir=./hparams/MALMEN/codellama-7b.yaml \
#     --data_set=EditCodeSearchNet > logs/malmen_cs_codellama2.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=MALMEN \
#     --data_dir ./[your splitted data dir] \
#     --hparams_dir=./hparams/MALMEN/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/malmen_cs_CodeQwen2.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=MALMEN \
#     --data_dir ./[your splitted data dir] \
#     --hparams_dir=./hparams/MALMEN/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet > logs/malmen_cs_stablecode_loo_2.log &


### ROME
# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=ROME \
#     --hparams_dir=./hparams/ROME/codellama-7b.yaml \ > logs/rome_cs_codellama.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=ROME \
#     --hparams_dir=./hparams/ROME/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/rome_cs_CodeQwen.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=ROME \
#     --hparams_dir=./hparams/ROME/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet> logs/rome_cs_stablecode.log &


### MEMIT
# CUDA_VISIBLE_DEVICES=2 nohup python edit_main.py \
#     --editing_method=MEMIT \
#     --hparams_dir=./hparams/MEMIT/codellama-7b.yaml > logs/memit_cs_codellama.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=MEMIT \
#     --hparams_dir=./hparams/MEMIT/CodeQwen1.5-7B.yaml \
#     --data_set=EditCodeSearchNet > logs/memit_cs_CodeQwen.log &

# CUDA_VISIBLE_DEVICES=4 nohup python edit_main.py \
#     --editing_method=MEMIT \
#     --hparams_dir=./hparams/MEMIT/stable-code-3b.yaml \
#     --data_set=EditCodeSearchNet> logs/memit_cs_stablecode.log &


### AGRACE
# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=AGRACE \
#     --hparams_dir=./hparams/AGRACE/codellama-7b.yaml \
#     --data_dir ./[your splitted data dir]\
#     --data_set=EditCodeSearchNet > logs/agrace_cs_codellama_loo2_wo_mean.log &

# CUDA_VISIBLE_DEVICES=5 nohup python edit_main.py \
#     --editing_method=AGRACE \
#     --hparams_dir=./hparams/AGRACE/CodeQwen1.5-7B.yaml \
#     --data_dir ./[your splitted data dir] \
#     --data_set=EditCodeSearchNet > logs/agrace_cs_codeqwen_loo2_wo_mean.log &

# CUDA_VISIBLE_DEVICES=7 nohup python edit_main.py \
#     --editing_method=AGRACE \
#     --hparams_dir=./hparams/AGRACE/stable-code-3b.yaml \
#     --data_dir ./[your splitted data dir] \
#     --data_set=EditCodeSearchNet > logs/agrace_cs_stable-code-3b_loo2_wo_mean.log &