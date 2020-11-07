import torch
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from src.efficientnet import EfficientNet


class AnnotatedRegionsDataset(Dataset):

    def __init__(self, annotation_cvs_path, image_path, transform=None):
        self._df = pd.read_csv(annotation_cvs_path)
        self._img_path = Path(image_path)
        self.transform = transform

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self._img_path / f"{self._df.iloc[idx, 0]}.jpeg"
        image = Image.open(img_name)

        sample = {'image': image, 'cls': self._df.iloc[idx, 3]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def train():
    data_path = Path('/home/sheh/datasets/TissueNet')
    dataset = AnnotatedRegionsDataset(
        data_path / "train_annotations.csv",
        data_path / "annotated_regions",
        transform=transforms.Compose([
            transforms.Resize(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    )
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    net = EfficientNet.from_pretrained('efficientnet-b0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["image"], data["cls"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == '__main__':
    train()
