CODE="$HOME/School/amir-security/code-authorship"
RESULTS="$HOME/School/amir-security/code-authorship/experiments/results"
RESULTS_PATH="${RESULTS}/largescale_300_table3.txt"

cd $CODE

N_ESTIMATORS="300"
MAXCLASSES_LST=(50 100 150 300 500 1000 1500 2000)

echo begin > $RESULTS_PATH

for MAXCLASSES in "${MAXCLASSES_LST[@]}"
do
    python codeauthorship/scripts/train_multilang.py \
        --path_py ~/Downloads/gcj-py-table3.jsonl \
        --n_estimators $N_ESTIMATORS \
        --max_classes $MAXCLASSES \
        --max_features 2500 \
        --json_result \
        >> $RESULTS_PATH 2>&1
done
