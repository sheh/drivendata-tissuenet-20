from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm

from src.epth_classifier import EpthClassifier
from src.region_classifier import RegionClassifier
from src.slide import UnlabeledSlidesListLocalTrain, SlidesListLocalTrain
from src.tools import BBox, best_page_for_sz


def main(args):
    clf = RegionClassifier(model_path=args.clf_model_path)
    CLASS_MIN_PROB = [0.97, 0.8, 0.9, 0.92]
    if args.unlabeled:
        #df_file_name = "manual_annotations_unlabeled.csv"
        df_file_name = "high_score_roi_unlabeled.csv"
        slides = UnlabeledSlidesListLocalTrain()
    else:
        df_file_name = "manual_annotations.csv"
        slides = SlidesListLocalTrain()

    df = pd.read_csv(df_file_name)

    pseudo_class = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        slide = slides.get_by_name(row.filename)
        clf_page_n = best_page_for_sz(slide.annotation_size(0), clf.image_size)
        bb = BBox(row.x1, row.y1, row.x2, row.y2, page=0).with_page(clf_page_n)
        img, _ = slide.get_page_region(crop_bb=bb, pad=True)
        probs = clf.predict_softmax([img])[0]
        cls = probs.argmax()
        max_prob = probs[cls]
        cls = cls if max_prob >= CLASS_MIN_PROB[cls] else -1
        if max_prob >= CLASS_MIN_PROB[cls] and args.show:
            img = cv2.resize(img, (0, 0), fx=4, fy=4)
            cv2.putText(img, str(probs), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
            cv2.imshow("", img)
            cv2.waitKey()
        pseudo_class.append(cls)
    df["pseudo_class"] = pseudo_class
    df["pseudo_class"] = df["pseudo_class"].astype(int)
    print(f"Total fits: {len(df[df['pseudo_class'] >= 0])}")
    print(df.groupby("pseudo_class").count())
    df.to_csv(args.output_file_name, index=False)
    print(f"Output here: {args.output_file_name}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clf-model-path", default="./logdir-clf/checkpoints/best_full.pth")
    parser.add_argument("--output-file-name")
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--show", action="store_true")
    #parser.add_argument("--min-prob", default=0.97, type=float)

    main(parser.parse_args())
