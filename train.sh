#!/bin/bash

RVERB="-v --dry-run"
RVERB=""
DISTSUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAVE_DIR="$(
python - "$@" << END
if 1:
  import train
  parser = train.get_parser()
  args = parser.parse_args()
  print(args.save_dir)
END
)"

mkdir -p ${SAVE_DIR}/code
rsync --exclude '.*' \
      --exclude data \
      --exclude pretrained_models \
      --exclude '*runs*' \
      --exclude '*.pyc' \
      --exclude '*.ipynb' \
      --filter=':- .gitignore' \
    $RVERB -lrpt $DISTSUP_DIR/ ${SAVE_DIR}/code/

echo $0 "$@" >> ${SAVE_DIR}/out.txt
exec python -u train.py "$@" 2>&1 | tee -ai ${SAVE_DIR}/out.txt
