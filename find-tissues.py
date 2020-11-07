import cv2
from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from src.slide import SlidesListLocalTrain, Slide, UnlabeledSlidesListLocalTrain
from src.tools import BBox


def test_slide_get_tissue():
    slides = SlidesListLocalTrain()
    missed_tissue_n = 0
    found_tissues = 0
    tissues_with_annotations = 0
    cv2.namedWindow("tissue", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("tissue", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("tissue", cv2.WINDOW_GUI_NORMAL)

    for slide in tqdm(slides):
        tissues = slide.get_tissues(max_locs=12, pad=0.1)
        # if tissues:
        #     continue

        img, page = slide.get_page_region(page_n=6)

        for i, t in enumerate(tissues):
            bb = t.bb.with_page(page)
            cv2.rectangle(img, bb.pt1, bb.pt2, (255, 0, 0), 3)
            # img, _ = slide.get_page_region(crop_bb=t.bb.with_page(7))
        cv2.putText(img, str(slide.label), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 3)
        cv2.imshow("tissue", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey()
        if k == 27:
            break

        missed_tissue_n += int(len(tissues) == 0)
        found_tissues += len(tissues)
        tissues_with_annotations += sum([1 for t in tissues if t.has_annotations])
    print("missed_tissue_n ", missed_tissue_n)
    print("found_tissues ", found_tissues)
    print(f"avg tissues {found_tissues/len(slides):.1f}")
    print(f"tissues with annotations {tissues_with_annotations / found_tissues:.1f}")


def _contours2rects(contours, pad, width, height, ratio_limit=7.0, min_area=0.1):
    # calc bounding boxes
    rects = sorted(map(cv2.boundingRect, contours), key=lambda x: x[2] * x[3], reverse=True)
    # filter by ration
    rects = list(filter(lambda x: (1 / ratio_limit) < x[2] / x[3] < (ratio_limit / 1), rects))
    # filter by area
    rects = list(filter(lambda x: x[2] * x[3] / (rects[0][2] * rects[0][3]) > min_area, rects))
    # add padding and convert to x1, y1, x2, y2
    return list(map(lambda x: BBox(
        max(int(x[0] - pad * x[2]), 0),
        max(int(x[1] - pad * x[3]), 0),
        min(int(x[0] + pad * x[2] + x[2]), width),
        min(int(x[1] + pad * x[3] + x[3]), height),
    ), rects))


def dbg_slide_get_tissue():
    white_pad = 0.05
    white_column_lvl = 254
    black_column_lvl = -1
    column_min_width = 0.05

    #slides = SlidesListLocalTrain()
    slides = UnlabeledSlidesListLocalTrain()
    for slide in tqdm(slides):
        img, page = slide.get_page_region(page_n=8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove white frame
        gray_dilate = cv2.dilate(gray, np.ones((15, 15), np.uint8), iterations=1)
        y1_w, x1_w = np.argwhere(gray_dilate != 255).min(axis=0)
        y2_w, x2_w = np.argwhere(gray_dilate != 255).max(axis=0)
        gray = gray[y1_w:y2_w, x1_w:x2_w]

        # remove white columns
        gray, offsets = Slide.remove_vertical_column(gray, white_column_lvl, column_min_width)
        gray_removed_columns = gray.copy()

        # remove black lines
        thresh_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
        gray[thresh_black == 255] = np.bincount(gray.ravel()).argmax()
        gray_without_blk = gray.copy()

        # find contours
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        gray = (255 * (gray / gray.max())).astype(np.uint8)
        gray_scales = gray.copy()
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = Slide._find_contours(gray)
        cv2.drawContours(gray, contours, -1, 0, 5)

        possible_tissue_bbs = _contours2rects(contours, 0.1, gray.shape[1], gray.shape[0])
        for bb in possible_tissue_bbs:
            bb = bb.with_offset(x1_w + offsets[bb.x1], y1_w)
            cv2.rectangle(img, bb.pt1, bb.pt2, (255, 0, 0), 3)
        #if not possible_tissue_bbs:
        fig, axes = plt.subplots(3, 2)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"original {slide.name}")
        axes[0, 1].imshow(gray_dilate, cmap='gray')
        axes[0, 1].set_title("gray_dilate")

        axes[1, 0].imshow(gray_without_blk, cmap='gray')
        axes[1, 0].set_title("gray_without_blk")
        axes[1, 1].imshow(gray_removed_columns, cmap='gray')
        axes[1, 1].set_title("gray_removed_columns")

        axes[2, 0].imshow(thresh, cmap='gray')
        axes[2, 0].set_title("thresh")
        axes[2, 1].imshow(gray_scales, cmap='gray')
        axes[2, 1].set_title("gray_scales")

        plt.show()


def main():
    dbg_slide_get_tissue()


if __name__ == '__main__':
    main()
