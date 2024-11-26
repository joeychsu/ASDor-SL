
for model_name in BEATs_iter3_plus_AS2M ; do
    for hidden_dim in 768 ; do
        projection_type="JustMeanProjection"
        results_path="results/${model_name}_v3_${projection_type}"
        #results_path="results/${model_name}_v3_${projection_type}_aug05_e3000"
        mkdir -p $results_path
        python local/split_annotations_cli.py --train_ratio 0.8 "ASDor_wav/train_annotations_v3.csv" \
            "${results_path}/tr_annotations.csv" "${results_path}/cv_annotations.csv"
        python local/train.py --projection_type "JustMeanProjection" \
            --batch_size 64 --num_epochs 3 --hidden_dim ${hidden_dim} \
            --model_path "pretrained_models/${model_name}.pt" \
            --seed 905 --freeze_encoder \
            --train_csv "${results_path}/tr_annotations.csv" \
            --valid_csv "${results_path}/cv_annotations.csv" --output_dir ${results_path}
        # --freeze_encoder --initial_lr 0.0 --final_lr 0.0 --use_augmentation 
    done
done

