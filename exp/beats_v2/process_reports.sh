#!/bin/bash

process_file() {
    local file=$1
    local model=$(basename $(dirname "$file"))
    
    awk -v model="$model" '
    BEGIN { OFS="\t" }
    /accuracy/ {
        accuracy = $2
    }
    /ok|ng|other/ {
        gsub(/[[:space:]]+/, " ")
        split($0, a, " ")
        class = a[1]
        precision[class] = a[2]
        recall[class] = a[3]
        f1[class] = a[4]
    }
    END {
        printf "%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
            model, 
            accuracy,
            precision["ok"], recall["ok"], f1["ok"],
            precision["ng"], recall["ng"], f1["ng"],
            precision["other"], recall["other"], f1["other"]
    }
    ' "$file"
}

echo -e "模型名稱\t整體準確率\tok精確度\tok召回率\tokF1\tng精確度\tng召回率\tngF1\tother精確度\tother召回率\totherF1"

for file in results/*/classification_report.txt; do
    process_file "$file"
done | sort -k1n