from typing import List

import torch
import logging
from src.efficientnet import EfficientNet
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage


class RegionClassifier:

    def __init__(self, model_path):
        num_classes = 4
        state_dict = torch.load(model_path)
        self.image_size = state_dict["checkpoint_data"]["image_size"]
        model_name = state_dict["checkpoint_data"]["network_type"]
        self._pre_processing = Compose([
            ToPILImage(),
            Resize((self.image_size, self.image_size)),
            ToTensor(),
            Normalize(mean=(0.73108001, 0.54549926, 0.67233236), std=(0.11515842, 0.13832434, 0.11472176)),
        ])
        self.model = EfficientNet.from_name(model_name=model_name, num_classes=num_classes)
        ret = self.model.load_state_dict(state_dict["model_state_dict"])
        assert not set(ret.missing_keys), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        print(f"Clf val metrics: {state_dict['valid_metrics']}")
        self.device = torch.device("cuda")
        self.model.eval()
        self.model.to(self.device)

    def predict_softmax(self, batch):
        batch = list(map(lambda x: self._pre_processing(x), batch))
        model_input = torch.stack(batch, dim=0)
        with torch.no_grad():
            logits = self.model(model_input.to(self.device))
            prob = logits.softmax(1).detach().cpu().numpy()
        return prob

    def predict_class(self, batch):
        return self.predict_softmax(batch).argmax(axis=1)
