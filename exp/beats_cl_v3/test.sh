
result_name_list="                    BEATs_iter3_plus_AS2M_v3_ConvProjection                   BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e1000"
result_name_list="${result_name_list} BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e2000       BEATs_iter3_plus_AS2M_v3_ConvProjection_e1000"
result_name_list="${result_name_list} BEATs_iter3_plus_AS2M_v3_ConvProjection_e2000             BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e3000"
result_name_list="${result_name_list} BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e1000_hd512 BEATs_iter3_plus_AS2M_v3_ConvProjection_aug05_e1000_hd1024"
result_name_list="BEATs_iter3_plus_AS2M_v3_ConvProjection"

n_enrollment=0
test_sets="test_annotations_v3_1 test_annotations_v3_2 test_annotations_v3_3 test_annotations_v3"
test_sets="test_annotations_v3_2"
:<<BLOCK
for result_name in ${result_name_list} ; do
    results_path="results/${result_name}"
    python local/extract_class_vector.py --model_path ${results_path}/best_model.pth \
      --test_csv ${results_path}/cv_annotations.csv \
      --output_dir ${results_path}
done
# --use_cpu
BLOCK
for result_name in ${result_name_list} ; do
    results_path="results/${result_name}"
    for test_set in ${test_sets} ; do
        echo -n "" > ${results_path}/${test_set}_total_classification_summary.txt
        for random_seed in `seq 901 902` ; do
            python local/test.py --seed ${random_seed} --ng_scale 1.6 \
              --class_vector_pt ${results_path}/enroll_vectors.pt --prefix "${test_set}" \
                --model_path ${results_path}/best_model.pth --n_enrollment ${n_enrollment} \
                --test_csv "ASDor_wav/${test_set}.csv" --output_dir ${results_path}
            cat ${results_path}/${test_set}_classification_summary.txt >> ${results_path}/${test_set}_total_classification_summary.txt
        done
        # --freeze_encoder --initial_lr 0.0 --final_lr 0.0 --use_augmentation --use_cpu 
    done
done

for result_name in ${result_name_list} ; do
    results_path="results/${result_name}"
    for test_set in ${test_sets} ; do
        # 初始化累加器
        sum_all_acc=0
        sum_quality=0
        sum_intra_class=0
        sum_inter_class=0
        sum_c0_acc=0
        sum_c1_acc=0
        sum_c2_acc=0
        sum_class_avg=0
        row_count=0

        # 定義輸入文件
        input_file="${results_path}/${test_set}_total_classification_summary.txt"
        printf "%s\n" "==============================================================================================================================="
        printf "Convert ${input_file} to Table ... \n\n"
        printf "%-7s %-7s %-7s %-7s %-11s %-11s %-11s %-11s %-11s %-11s %-11s %-11s\n" \
               "TestID" "Enroll" "Class" "Samples" "All Acc" "Quality" "Intra-class" \
               "Inter-class" "C0 Acc" "C1 Acc" "C2 Acc" "Class Avg"
        printf "%s\n" "-------------------------------------------------------------------------------------------------------------------------------"
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

            # 累加到總和
            sum_all_acc=$(awk -v sum="$sum_all_acc" -v val="$overall_accuracy" 'BEGIN {print sum + val}')
            sum_quality=$(awk -v sum="$sum_quality" -v val="$quality_ratio" 'BEGIN {print sum + val}')
            sum_intra_class=$(awk -v sum="$sum_intra_class" -v val="$intra_class_similarity" 'BEGIN {print sum + val}')
            sum_inter_class=$(awk -v sum="$sum_inter_class" -v val="$inter_class_similarity" 'BEGIN {print sum + val}')
            sum_c0_acc=$(awk -v sum="$sum_c0_acc" -v val="$class_0_accuracy" 'BEGIN {print sum + val}')
            sum_c1_acc=$(awk -v sum="$sum_c1_acc" -v val="$class_1_accuracy" 'BEGIN {print sum + val}')
            sum_c2_acc=$(awk -v sum="$sum_c2_acc" -v val="$class_2_accuracy" 'BEGIN {print sum + val}')
            sum_class_avg=$(awk -v sum="$sum_class_avg" -v val="$class_avg_accuracy" 'BEGIN {print sum + val}')
            row_count=$((row_count + 1))

            # 將測試結果顯示在 Console
            printf "%-7s %-7s %-7s %-7s %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f\n" \
                   "$test_id" "$enrollment_samples" "$classes" "$test_samples" "$overall_accuracy" \
                   "$quality_ratio" "$intra_class_similarity" "$inter_class_similarity" "$class_0_accuracy" \
                   "$class_1_accuracy" "$class_2_accuracy" "$class_avg_accuracy"

            test_id=$((test_id + 1))  # 測試編號加1
          fi
        done < "$input_file"
        # 計算平均值
        avg_all_acc=$(awk -v sum="$sum_all_acc" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_quality=$(awk -v sum="$sum_quality" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_intra_class=$(awk -v sum="$sum_intra_class" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_inter_class=$(awk -v sum="$sum_inter_class" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_c0_acc=$(awk -v sum="$sum_c0_acc" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_c1_acc=$(awk -v sum="$sum_c1_acc" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_c2_acc=$(awk -v sum="$sum_c2_acc" -v count="$row_count" 'BEGIN {print sum / count}')
        avg_class_avg=$(awk -v sum="$sum_class_avg" -v count="$row_count" 'BEGIN {print sum / count}')

        # 顯示平均值行
        printf "%-7s %-7s %-7s %-7s %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f\n" \
               "AVG" "-" "-" "-" "$avg_all_acc" "$avg_quality" "$avg_intra_class" "$avg_inter_class" \
               "$avg_c0_acc" "$avg_c1_acc" "$avg_c2_acc" "$avg_class_avg"
        printf "%s\n" "==============================================================================================================================="

        echo
    done
done

