from typing import List

from src.tools import BBox, Annotation


class Tissue:

    def __init__(self, bb: BBox, annotations: List[Annotation]):
        self.bb = bb
        self.annotations = annotations

    def get_bb_page(self, page):
        return self.bb.with_page(page)

    def get_relative_annotations_page(self, page):
        bb = self.bb.with_page(page)
        return list(map(lambda x: x.with_page(page).with_offset(-bb.x1, -bb.y1), self.annotations))

    @property
    def has_annotations(self):
        return len(self.annotations)
