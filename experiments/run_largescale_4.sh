CODE="$HOME/School/amir-security/code-authorship"
RESULTS="$HOME/School/amir-security/code-authorship/experiments/results"
RESULTS_PATH="${RESULTS}/largescale_4.txt"

cd $CODE

N_ESTIMATORS="4"
MAXCLASSES_LST=(50 100 150 300 500 1000 1500 2000)

echo begin > $RESULTS_PATH

for MAXCLASSES in "${MAXCLASSES_LST[@]}"
do
    python codeauthorship/scripts/train_multilang.py \
        --path_py ~/Downloads/gcj-py-small.jsonl \
        --n_estimators $N_ESTIMATORS \
        --max_classes $MAXCLASSES \
        --json_result \
        >> $RESULTS_PATH 2>&1
done
