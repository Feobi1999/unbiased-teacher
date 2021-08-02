#!/bin/bash
cd baseline
for file in $(ls *)
do
  echo $file
done


#!/bin/bash
SPLIT=(10 12 8 6 4 2 13 14 15 16 17 18 19 20 21 22 23 24)

#a=3000
#for i in ${SPLIT[*]}
#do
#  c=`expr $i \* $a`
#  python3 predict.py \
#      --evaluate ${PSEUDO_PATH}/eval.json \
#      --load "${CKPT_PATH}"/model-$c \
#      --config \
#      DATA.BASEDIR=${COCODIR} \
#      DATA.TRAIN="('${UNLABELED_DATASET}',)" | tee log_${i}.txt
#done