#### Conala
### CodeLlama
# CUDA_VISIBLE_DEVICES=0 nohup python train_agrace_encoder.py \
#     --model=../ptms/CodeLlama-7b-hf \
#     --data_set=EditConala \
#     --mode=mean > logs/train_agrace_encoder_codellama_mean.log &

# CUDA_VISIBLE_DEVICES=2 nohup python train_agrace_encoder.py \
#     --model=../ptms/CodeQwen1.5-7B \
#     --data_set=EditConala \
#     --mode=mean > logs/train_agrace_encoder_codeqwen_mean.log &

# CUDA_VISIBLE_DEVICES=2 nohup python train_agrace_encoder.py \
#     --model=../ptms/stable-code-3b \
#     --data_set=EditConala \
#     --mode=mean > logs/train_agrace_encoder_stablecode_mean.log &

# CUDA_VISIBLE_DEVICES=4 nohup python train_agrace_encoder.py \
#     --model=../ptms/CodeLlama-7b-hf \
#     --data_set=EditCodeSearchNet \
#     --mode=mean > logs/train_agrace_encoder_cs_codellama_mean.log &

# CUDA_VISIBLE_DEVICES=5 nohup python train_agrace_encoder.py \
#     --model=../ptms/CodeQwen1.5-7B \
#     --data_set=EditCodeSearchNet \
#     --mode=mean > logs/train_agrace_encoder_cs_codeqwen_mean.log &

# CUDA_VISIBLE_DEVICES=3 nohup python train_agrace_encoder.py \
#     --model=../ptms/stable-code-3b \
#     --data_set=EditCodeSearchNet \
#     --mode=mean > logs/train_agrace_encoder_cs_stablecode_mean.log &

models=( "../ptms/stable-code-3b" "../ptms/CodeLlama-7b-hf" "../ptms/CodeQwen1.5-7B" )
data_sets=( "EditConala" "EditCodeSearchNet" )
data_dirs=( "./[your splited data dir1]" "./[your splited data dir2]" )
save_dirs=( "./results/[your splitted data dir1]" "./results/[your splitted data dir1]" )

# 遍历所有组合
for model in "${models[@]}"
do
    for data_set in "${data_sets[@]}"
    do
        for i in "${!data_dirs[@]}"
        do
            data_dir="${data_dirs[$i]}"
            save_dir="${save_dirs[$i]}"

            echo "Running with model: $model, dataset: $data_set, data_dir: $data_dir, save_dir: $save_dir, mode: last_token"
            
            CUDA_VISIBLE_DEVICES=3 python train_agrace_encoder.py --model "$model" --data_set "$data_set" --data_dir "$data_dir" --save_dir "$save_dir" --mode last_token
        done
    done
done