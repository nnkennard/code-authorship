CAHOME=${CAHOME:-$HOME/School/amir-security/code-authorship}
RESULTS="$CAHOME/experiments/results"
CADATA=${CADATA:-$HOME/Downloads}

RESULTS_PATH=${RESULTS}/ablation_include.txt

echo $CAHOME
echo $RESULTS
echo $CADATA

cd $CAHOME

TOKEN_TYPES=(
'OP'
'NAME'
'NEWLINE'
'NUMBER'
'INDENT'
'DEDENT'
'NL'
'STRING'
'COMMENT'
'ENCODING'
'ENDMARKER'
'ERRORTOKEN'
)

DATAPATH=$CADATA/gcj-py-2014.jsonl

echo begin > $RESULTS_PATH

for TT1 in "${TOKEN_TYPES[@]}"
do
    for TT2 in "${TOKEN_TYPES[@]}"
    do
        echo $TT1 $TT2
        python codeauthorship/scripts/train_multilang.py \
            --include_type $TT1,$TT2 \
            --path_py $DATAPATH \
            --n_estimators 300 --max_features 2500 --json_result >> $RESULTS_PATH  2>&1
    done
done
