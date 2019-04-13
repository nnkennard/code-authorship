CODE="/Users/adrozdov/School/amir-security/code-authorship"
RESULTS="/Users/adrozdov/School/amir-security/code-authorship/experiments/results"
RESULTS_PATH="${RESULTS}/onlyfeature.txt"

cd $CODE

TOKEN_TYPES=(
'comment'
'string'
'newline'
'number'
'indent' 'dedent' 'encoding' 'endmarker'
'errortoken' 'nl' 'name' 'op'
)

echo begin > $RESULTS_PATH

for TT1 in "${TOKEN_TYPES[@]}"
do
    for TT2 in "${TOKEN_TYPES[@]}"
    do
        echo $TT1 $TT2
        python codeauthorship/scripts/train-baseline.py --only$TT1 --only$TT2 --json_result >> $RESULTS_PATH
    done
done
