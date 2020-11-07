#!/bin/bash -ex

#TISSIENET_TIF=$1
#TISSIENET_CSV=$2
if test -z "$TISSIENET_TIF" && test -z "$TISSIENET_CSV"
then
      echo "\$TISSIENET_TIF, \$TISSIENET_CSV \$TISSIENET_TIF_UNLABELED are not set"
      exit 1
fi

CSV_PATH=./inference-data
PAD_CLF=0.1
PAD_ETH=0

mkdir -p "${CSV_PATH}"

python scripts/train-test-split.py "${TISSIENET_CSV}" "${CSV_PATH}" --folds-n 4 --seed 42

# train epithelium classifier using the annotations and the custom labeling
python train_clf_epithelium.py --tif-path "${TISSIENET_TIF}" --csv-path "${CSV_PATH}" \
  --num-epoch 70 \
  --image-size 128 \
  --batch-size 128 \
  --lr 0.02 \
  --pad "${PAD_ETH}" \
  --network-type efficientnet-b0 \
  --not-use-unlabeled

# train classifier using the annotations
python train_clf.py --tif-path "${TISSIENET_TIF}" --csv-path "${CSV_PATH}" \
  --num-epochs 120 \
  --image-size 192 \
  --batch-size 96 \
  --lr 0.01 \
  --pad "${PAD_CLF}" \
  --network-type efficientnet-b0

# get high scored regions from unlabeled data
python ./pseudo_annotate_unlabeled.py

# automatic annotation using the regions from previuos step
python ./pseudo_annotate.py --unlabeled --output-file-name ./pseudo-clf.csv

# train a clf using automatic labeled data as training dataset and the original annotations as test dataset
python train_clf.py --tif-path "${TISSIENET_TIF}" --csv-path "${CSV_PATH}" \
  --log-dir ./logdir-clf-pseudo \
  --num-epochs 120 \
  --image-size 192 \
  --batch-size 96 \
  --lr 0.01 \
  --pad "${PAD_CLF}" \
  --network-type efficientnet-b0 \
  --use-pseudo-annotations


# fine tune using original labeling
python train_clf.py --tif-path "${TISSIENET_TIF}" --csv-path "${CSV_PATH}" \
  --log-dir ./logdir-clf-ftune \
  --num-epochs 120 \
  --image-size 192 \
  --batch-size 96 \
  --lr 0.0001 \
  --pad "${PAD_CLF}" \
  --network-type efficientnet-b0 \
  --resume logdir-clf-pseudo/checkpoints/best_full.pth

mkdir -p ./models
cp ./logdir-clf-epithelium/checkpoints/best_full.pth ./models/epth.pth
cp ./logdir-clf-ftune/checkpoints/best_full.pth ./models/clf.pth

# train random forest classifier and score train set
python main.py --tif-path "${TISSIENET_TIF}" \
  --epth-model-path models/epth.pth \
  --clf-model-path models/clf.pth \
  --metadata-path "${CSV_PATH}"/train_metadata.csv \
  --submission-format-path "${CSV_PATH}"/train_submission_format.csv \
  --annotations-path "${CSV_PATH}"/train_annotations.csv \
  --labels-path "${CSV_PATH}"/train_labels.csv \
  --log-path main-logs/submission.log \
  --roi-probs-path "./train_roi_probs.csv" \
  --train-random-forest \
  --model-random-forest ./models/rf-clf.pkl \
  --local-test

# score test set
python main.py --tif-path "${TISSIENET_TIF}" \
  --epth-model-path models/epth.pth \
  --clf-model-path models/clf.pth \
  --metadata-path "${CSV_PATH}"/test_metadata.csv \
  --submission-format-path "${CSV_PATH}"/submission_format.csv \
  --annotations-path "${CSV_PATH}"/test_annotations.csv \
  --labels-path "${CSV_PATH}"/test_labels.csv \
  --log-path main-logs/submission.log \
  --roi-probs-path "./test_roi_probs.csv" \
  --model-random-forest ./models/rf-clf.pkl \
  --local-test

