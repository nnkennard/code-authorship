CODE="/Users/adrozdov/School/amir-security/code-authorship"
RESULTS="/Users/adrozdov/School/amir-security/code-authorship/experiments/results"
RESULTS_PATH="${RESULTS}/minthreshold.txt"

cd $CODE

THRESHOLD_LST=(
50
100
150
200
250
300
350
400
450
500
)

echo begin > $RESULTS_PATH

for THRESHOLD in "${THRESHOLD_LST[@]}"
do
    ENAME="onlyreserved-t_${THRESHOLD}"
    echo $ENAME
    python codeauthorship/scripts/train_baseline.py --onlyname --onlyreserved \
        --minthreshold $THRESHOLD --json_result --name $ENAME \
        >> $RESULTS_PATH
done

for THRESHOLD in "${THRESHOLD_LST[@]}"
do
    ENAME="noreserved-t_${THRESHOLD}"
    echo $ENAME
    python codeauthorship/scripts/train_baseline.py --onlyname --noreserved \
        --minthreshold $THRESHOLD --json_result --name $ENAME \
        >> $RESULTS_PATH
done

for THRESHOLD in "${THRESHOLD_LST[@]}"
do
    ENAME="both-t_${THRESHOLD}"
    echo $ENAME
    python codeauthorship/scripts/train_baseline.py --onlyname \
        --minthreshold $THRESHOLD --json_result --name $ENAME \
        >> $RESULTS_PATH
done
