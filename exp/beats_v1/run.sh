
model_list="              BEATs_iter1 BEATs_iter2 BEATs_iter3"
model_list="${model_list} BEATs_iter3_plus_AS20K BEATs_iter3_plus_AS2M"

for model_name in ${model_list} ; do
	#python local/train.py --batch_size 8 --num_epochs 100 \
    #	--model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
    #	--freeze_encoder --use_augmentation --seed 905 \
    #	--output_dir "results/${model_name}_fixencoder"
    python local/test.py --model_path "results/${model_name}_fixencoder/best_model.pth" \
    	--test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}_fixencoder"
done

for model_name in ${model_list} ; do
	#python local/train.py --batch_size 8 --num_epochs 100 \
    #	--model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
    #	--use_augmentation --seed 905 \
    #	--output_dir "results/${model_name}"
	python local/test.py --model_path "results/${model_name}/best_model.pth" \
    	--test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}"
done

model_name=BEATs_iter3_plus_AS2M
for seed in 905 906 907 908 909 910 911 912 913 914 ; do
	#python local/train.py --batch_size 8 --num_epochs 100 \
    #	--model_path "pretrained_models/${model_name}.pt" --num_labels 3 \
    #	--use_augmentation --seed ${seed} \
    #	--output_dir "results/${model_name}_seed${seed}"
	python local/test.py --model_path "results/${model_name}_seed${seed}/best_model.pth" \
    	--test_csv ASDor_wav/test_annotations.csv --output_dir "results/${model_name}_seed${seed}"
done
