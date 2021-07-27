# predictors=(
# fisher grad_norm grasp jacov snip synflow \
# lce lce_m sotl sotle valacc valloss \
# lcsvr 
predictors=(bohamiann)

# experiment_types=(single single single single single single \
# vary_fidelity vary_fidelity vary_fidelity vary_fidelity vary_fidelity vary_fidelity \
# vary_both 
experiment_types=(vary_train_size)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=1
fi

# folders:
base_file=/home/zabergjg/DL_LAB/nas_predictors/naslib
s3_folder=p201_im_SMAC
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=nasbench201
dataset=cifar10

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
save_to_s3=true
test_size=200
hpo_method=SMAC

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}
    python $base_file/benchmarks/create_configs.py --predictor $predictor --experiment_type $experiment_type \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --config_type predictor --search_space $search_space --hpo_method $hpo_method
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python $base_file/benchmarks/predictors/runner.py --config-file $config_file
    done
    if [ "$save_to_s3" ]
    then
        # zip and save to s3
        echo zipping and saving to s3
        zip -r $out_dir.zip $out_dir 
        python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir --s3_folder $s3_folder
    fi
done
