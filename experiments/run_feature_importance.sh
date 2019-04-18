
CODE="/Users/adrozdov/School/amir-security/code-authorship"
RESULTS="/Users/adrozdov/School/amir-security/code-authorship/experiments/results"

cd $CODE

RESULTS_PATH="${RESULTS}/feature_importance-both.txt"
python codeauthorship/scripts/train_baseline.py  --onlyname --json_result \
--include_feature_importance > $RESULTS_PATH

RESULTS_PATH="${RESULTS}/feature_importance-reserved.txt"
python codeauthorship/scripts/train_baseline.py  --onlyname --onlyreserved --json_result \
--include_feature_importance > $RESULTS_PATH

RESULTS_PATH="${RESULTS}/feature_importance-udf.txt"
python codeauthorship/scripts/train_baseline.py  --onlyname --noreserved --json_result \
--include_feature_importance > $RESULTS_PATH