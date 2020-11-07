The repo contains training code for the [TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/page/254/) competition.

The inference pipeline:

1. Detect a tissue on a slide using "classical" cv
2. Split the found tissue on a grid and predict probability of "epithelium" class for each cell
3. Predict one of four classes for high-scored cells
4. Predict whole slide class using RandomForest classifier and aggregated cell predictions  

Check `train.sh` to get the training steps.

### Train

```
export TISSIENET_TIF=<path-to-tif-files>
export TISSIENET_CSV=<path-to-csv-files>
export TISSIENET_TIF_UNLABELED=<path-to-unlabeled-tif-files>

./train.sh

```

`<path-to-csv-files>` contains `train_annotations.csv, train_labels.csv, train_metadata.csv, unlabeled_metadata.csv`.

The trained models metrics:

```buildoutcfg
Epth clf val metrics: {'loss': 0.15075473606172712, 'accuracy': 0.9612957509465712, 'precision': 0.948333166636426, 'recall': 0.9581144033196733, 'f1': 0.9505933058916757}
Clf val metrics: {'loss': 0.7078824208820116, 'accuracy': 0.777404169468729, 'main_metric': 0.94364492266308, 'precision': 0.777404169468729, 'recall': 0.777404169468729, 'f1': 0.7774041694687289}
```

The final metric:

```buildoutcfg
Final metric: 0.921
```