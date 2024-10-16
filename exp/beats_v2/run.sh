
model_list="BEATs_iter3_plus_AS20K BEATs_iter3_plus_AS2M"

:<<BLOCK
for model_name in ${model_list} ; do
	python local/train.py --batch_size 8 --num_epochs 100 \
    	--model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
    	--freeze_encoder --use_augmentation --seed 905 \
    	--output_dir "results/${model_name}_fixencoder"
    python local/test.py --model_path "results/${model_name}_fixencoder/best_model.pth" \
    	--test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}_fixencoder"
done
BLOCK

:<<BLOCK
for model_name in ${model_list} ; do
    python local/train.py --batch_size 8 --num_epochs 100 \
        --model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
        --seed 905 \
        --output_dir "results/${model_name}"
    python local/test.py --model_path "results/${model_name}/best_model.pth" \
        --test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}"
done

for model_name in ${model_list} ; do
	python local/train.py --batch_size 8 --num_epochs 100 \
    	--model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
    	--use_augmentation --seed 905 \
    	--output_dir "results/${model_name}_aug"
	python local/test.py --model_path "results/${model_name}_aug/best_model.pth" \
    	--test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}_aug"
done
BLOCK

:<<BLOCK
model_name=BEATs_iter3_plus_AS2M
for seed in 905 906 907 908 909 910 911 912 913 914 ; do
	python local/train.py --batch_size 8 --num_epochs 100 \
    	--model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
    	--use_augmentation --seed ${seed} \
    	--output_dir "results/${model_name}_seed${seed}"
	python local/test.py --model_path "results/${model_name}_seed${seed}/best_model.pth" \
    	--test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}_seed${seed}"
done
BLOCK

:<<BLOCK
for model_name in BEATs_iter3_plus_AS2M ; do
    results_path="results/${model_name}_fixencoder"
    python local/train.py --batch_size 8 --num_epochs 100 \
        --model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
        --freeze_encoder --seed 905 \
        --output_dir ${results_path}
    python local/test.py --model_path "${results_path}/best_model.pth" \
        --test_csv ASDor_wav/test_annotations.csv --output_dir ${results_path}
done
BLOCK

:<<BLOCK
aug_type_list="                 None                      time_stretching    pitch_shifting "
aug_type_list="${aug_type_list} dynamic_range_compression add_gaussian_noise adjust_volume  "
for model_name in BEATs_iter3_plus_AS2M ; do
    for aug_type in ${aug_type_list} ; do
        results_path="results/${model_name}_fixencoder_aug_${aug_type}"
        python local/train.py --batch_size 8 --num_epochs 100 \
            --model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
            --freeze_encoder --use_augmentation --augment_type "${aug_type}" --seed 905 \
            --output_dir ${results_path}
        python local/test.py --model_path "${results_path}/best_model.pth" \
            --test_csv ASDor_wav/test_annotations.csv --output_dir ${results_path}
    done
done
BLOCK


for model_name in BEATs_iter3_plus_AS2M ; do
    for hidden_dim in 64 128 256 512 ; do
        results_path="results/${model_name}_b8_e100_h${hidden_dim}"
        python local/train.py --batch_size 8 --num_epochs 100 --hidden_dim ${hidden_dim} \
            --model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
            --seed 905 \
            --output_dir ${results_path}
        python local/test.py --model_path "${results_path}/best_model.pth" \
            --test_csv ASDor_wav/test_annotations.csv --output_dir ${results_path}
    done
done

for model_name in BEATs_iter3_plus_AS2M ; do
    for augment_prob in 0.1 0.3 0.5 0.7 ; do
        results_path="results/${model_name}_aug_None_b8_e100_h512_ap${augment_prob}"
        python local/train.py --batch_size 8 --num_epochs 100 --hidden_dim 512 \
            --model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
            --use_augmentation --augment_prob ${augment_prob} --seed 905 \
            --output_dir ${results_path}
        python local/test.py --model_path "${results_path}/best_model.pth" \
            --test_csv ASDor_wav/test_annotations.csv --output_dir ${results_path}
    done
done
