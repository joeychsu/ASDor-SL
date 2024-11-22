
result_name_list="                    BEATs_iter3_plus_AS2M_v3_ConvProjection             BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e1000"
result_name_list="${result_name_list} BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e2000 BEATs_iter3_plus_AS2M_v3_ConvProjection_e1000"

test_sets="test_annotations_v3_2"

for result_name in ${result_name_list} ; do
    results_path="results/${result_name}"
    for test_set in ${test_sets} ; do
        echo -n "" > ${results_path}/${test_set}_total_classification_summary.txt
        for random_seed in `seq 901 910` ; do
            n_enrollment=5
            python local/test.py --use_cpu --seed ${random_seed} \
                --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
                --test_csv "ASDor_wav/${test_set}.csv" --output_dir ${results_path} --prefix "${test_set}"
            cat ${results_path}/${test_set}_classification_summary.txt >> ${results_path}/${test_set}_total_classification_summary.txt
        done
        # --freeze_encoder --initial_lr 0.0 --final_lr 0.0 --use_augmentation 
    done
done


for result_name in ${result_name_list} ; do
    results_path="results/${result_name}"
    for test_set in ${test_sets} ; do
        # 定義輸入文件
        input_file="${results_path}/${test_set}_total_classification_summary.txt"
        printf "%s\n" "========================================================================================================================================="
        printf "Convert ${input_file} to Table ... \n\n"
        printf "%-8s %-8s %-8s %-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
               "TestID" "Enroll" "Class" "Samples" "Overall Acc" "Quality" "Intra-class" \
               "Inter-class" "Class0 Acc" "Class1 Acc" "Class2 Acc" "Class Avg"
        printf "%s\n" "-----------------------------------------------------------------------------------------------------------------------------------------"
        # 解析輸入文件
        test_id=1  # 用於記錄每次測試的編號
        while read -r line; do
          # 跳過空行
          [[ -z $line ]] && continue

          if [[ $line == "Classification Summary Report"* ]]; then
            # 初始化變數
            enrollment_samples=""
            classes=""
            test_samples=""
            overall_accuracy=""
            quality_ratio=""
            intra_class_similarity=""
            inter_class_similarity=""
            class_0_accuracy=""
            class_1_accuracy=""
            class_2_accuracy=""
          elif [[ $line == "Number of enrollment samples per class:"* ]]; then
            enrollment_samples=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Total number of classes:"* ]]; then
            classes=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Total number of test samples:"* ]]; then
            test_samples=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Overall Accuracy:"* ]]; then
            overall_accuracy=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Quality Ratio:"* ]]; then
            quality_ratio=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Intra-class Similarity:"* ]]; then
            intra_class_similarity=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Inter-class Similarity:"* ]]; then
            inter_class_similarity=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Class 0: Accuracy"* ]]; then
            class_0_accuracy=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Class 1: Accuracy"* ]]; then
            class_1_accuracy=$(echo "$line" | awk '{print $NF}')
          elif [[ $line == "Class 2: Accuracy"* ]]; then
            class_2_accuracy=$(echo "$line" | awk '{print $NF}')

          fi
          if [[ -n "$overall_accuracy" && -n "$class_0_accuracy" && -n "$class_2_accuracy" ]]; then
            # 計算平均準確率
            class_avg_accuracy=$(awk -v acc0="$class_0_accuracy" -v acc1="$class_1_accuracy" -v acc2="$class_2_accuracy" \
                                 'BEGIN {print (acc0 + acc1 + acc2) / 3}')

            # 將測試結果顯示在 Console
            printf "\n%-8s %-8s %-8s %-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
                   "$test_id" "$enrollment_samples" "$classes" "$test_samples" "$overall_accuracy" \
                   "$quality_ratio" "$intra_class_similarity" "$inter_class_similarity" "$class_0_accuracy" \
                   "$class_1_accuracy" "$class_2_accuracy" "$class_avg_accuracy"

            test_id=$((test_id + 1))  # 測試編號加1
          fi
        done < "$input_file"
        printf "%s\n" "========================================================================================================================================="

        echo
    done
done

