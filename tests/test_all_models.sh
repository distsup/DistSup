#!/bin/bash

# Usage
# -----
# Run from any dir (after sourcing set-env.sh):
#   $ bash test_all_models.sh [pattern]
# A logfile will be generated under DistSup/ .

function test_run_model {

  LOG="$REPORT_FILE"
  YAML=$1
  EXP_DIR=`mktemp -d -p $ROOT_EXP_DIR`

  echo -n "Investigating "$YAML" ... "
  echo -n `date +"%F_%H:%M:%S "` >> "$LOG"
  echo -n " "$1" " >> "$LOG"

  python train.py --debug-skip-training $YAML $EXP_DIR \
    -m Trainer.num_epochs 1 Trainer.polyak_decay 0 \
    2> $EXP_DIR/train.err > $EXP_DIR/train.out

  if [ "$?" == 0 ]; then
    echo "[OK]"
    echo "[OK]" >> "$LOG"
  else
    echo "[FAILED]"
    echo "[FAILED]" >> "$LOG"
    echo -e "\nTemp exp dir: $EXP_DIR\n" >> "$LOG"
    cat $EXP_DIR/train.err >> "$LOG"
    echo "" >> "$LOG"
  fi
}

cd $DISTSUP_DIR

TIMESTAMP=`date +"%F_%H.%M.%S"`
export -f test_run_model
export REPORT_FILE=`pwd`/test_yaml_report_${TIMESTAMP}.txt
export ROOT_EXP_DIR=`mktemp -d`

echo "     _________________________________________ "
echo "    / Running test models!                    \\"
echo "    | Temp dir: $ROOT_EXP_DIR"
echo "    \\ To preserve logs, cancel with ^C .      /"
echo "     ----------------------------------------- "
echo "            \   ^__^                           "
echo "             \  (xx)\_______                   "
echo "                (__)\       )\/\               "
echo "                 U  ||--WWW |                  "
echo "                    ||     ||                  "
echo ""

PATTERN=$1
find egs/ -wholename "*$PATTERN*.yaml" -exec bash -c 'test_run_model "$0"' {} \;

rm -rf "$ROOT_EXP_DIR"
