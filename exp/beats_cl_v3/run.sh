
for model_name in BEATs_iter3_plus_AS2M ; do
    for hidden_dim in 256 ; do
        #results_path="results/${model_name}_v3_3_fixen_b32_e100_h${hidden_dim}"
        #results_path="results/${model_name}_v3_3_justproj"
        #results_path="results/${model_name}_v3_SimpleFewShot"
        results_path="results/${model_name}_v3_ConvProjection"
        mkdir -p $results_path
        python local/split_annotations_cli.py --train_ratio 0.8 "ASDor_wav/train_annotations_v3.csv" \
            "${results_path}/tr_annotations.csv" "${results_path}/cv_annotations.csv"
        python local/train.py --batch_size 64 --num_epochs 500 --hidden_dim ${hidden_dim} \
            --model_path "pretrained_models/${model_name}.pt" \
            --seed 905 --freeze_encoder \
            --train_csv "${results_path}/tr_annotations.csv" \
            --valid_csv "${results_path}/cv_annotations.csv" --output_dir ${results_path}
        n_enrollment=5
        python local/test.py --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
            --test_csv "${results_path}/tr_annotations.csv" --output_dir ${results_path} --prefix "tr_annotations"
        python local/test.py --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
            --test_csv "${results_path}/cv_annotations.csv" --output_dir ${results_path} --prefix "cv_annotations"
        python local/test.py --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
            --test_csv "ASDor_wav/test_annotations_v3_1.csv" --output_dir ${results_path} --prefix "test_annotations_v3_1"
        python local/test.py --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
            --test_csv "ASDor_wav/test_annotations_v3_2.csv" --output_dir ${results_path} --prefix "test_annotations_v3_2"
        python local/test.py --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
            --test_csv "ASDor_wav/test_annotations_v3_3.csv" --output_dir ${results_path} --prefix "test_annotations_v3_3"
        # --freeze_encoder --initial_lr 0.0 --final_lr 0.0
    done
done

