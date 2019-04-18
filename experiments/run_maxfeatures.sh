CODE="/Users/adrozdov/School/amir-security/code-authorship"
RESULTS="/Users/adrozdov/School/amir-security/code-authorship/experiments/results"
RESULTS_PATH="${RESULTS}/max_features.txt"

cd $CODE

THRESHOLD_LST=(
10
20
40
60
80
100
200
400
600
800
1000
2000
3000
4000
)

echo begin > $RESULTS_PATH

for THRESHOLD in "${THRESHOLD_LST[@]}"
do
    ENAME="both-t_${THRESHOLD}"
    echo $ENAME
    python codeauthorship/scripts/train_baseline.py --onlyname \
        --max_features $THRESHOLD --json_result --name $ENAME \
        >> $RESULTS_PATH
done

for THRESHOLD in "${THRESHOLD_LST[@]}"
do
    ENAME="onlyreserved-t_${THRESHOLD}"
    echo $ENAME
    python codeauthorship/scripts/train_baseline.py --onlyname --onlyreserved \
        --max_features $THRESHOLD --json_result --name $ENAME \
        >> $RESULTS_PATH
done

for THRESHOLD in "${THRESHOLD_LST[@]}"
do
    ENAME="noreserved-t_${THRESHOLD}"
    echo $ENAME
    python codeauthorship/scripts/train_baseline.py --onlyname --noreserved \
        --max_features $THRESHOLD --json_result --name $ENAME \
        >> $RESULTS_PATH
done
