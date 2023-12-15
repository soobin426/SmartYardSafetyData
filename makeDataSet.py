import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = Image.open(img_path).convert("RGB")
        label = open(label_path, "r").readline().strip().split(" ")
        label = [float(x) for x in label]

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터셋 디렉토리 설정
data_dir = "./data"
dataSet_dir = "./dataSet"
classes = ["M0", "M1", "B0", "B1", "H0", "H1"]

# 훈련, 검증, 테스트 비율
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# DataLoader에 사용될 변환
transform = transforms.Compose([transforms.ToTensor()])

# 클래스별로 데이터셋을 훈련, 검증, 테스트로 나누기
for class_name in classes:
    images_dir = os.path.join(data_dir, class_name, "Images")
    labels_dir = os.path.join(data_dir, class_name, "labels")

    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith(".jpg")]
    label_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith(".txt")]

    train_images, test_images, train_labels, test_labels = train_test_split(
        image_files, label_files, test_size=(valid_ratio + test_ratio), random_state=42
    )
    valid_images, test_images, valid_labels, test_labels = train_test_split(
        test_images, test_labels, test_size=test_ratio / (valid_ratio + test_ratio), random_state=42
    )

    # train, valid, test 디렉토리 생성
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataSet_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for directory in ["images", "labels"]:
            os.makedirs(os.path.join(split_dir, directory), exist_ok=True)

    # 데이터 복사
    for src, dst in zip(train_images, train_labels):
        copyfile(src, os.path.join(dataSet_dir, "train", "images", os.path.basename(src)))
        copyfile(dst, os.path.join(dataSet_dir, "train", "labels", os.path.basename(dst)))

    for src, dst in zip(valid_images, valid_labels):
        copyfile(src, os.path.join(dataSet_dir, "valid", "images", os.path.basename(src)))
        copyfile(dst, os.path.join(dataSet_dir, "valid", "labels", os.path.basename(dst)))

    for src, dst in zip(test_images, test_labels):
        copyfile(src, os.path.join(dataSet_dir, "test", "images", os.path.basename(src)))
        copyfile(dst, os.path.join(dataSet_dir, "test", "labels", os.path.basename(dst)))

    print(f"Dataset for {class_name} is split and saved.")

    # DataLoader 생성
    train_dataset = CustomDataset(train_images, train_labels, transform=transform)
    valid_dataset = CustomDataset(valid_images, valid_labels, transform=transform)
    test_dataset = CustomDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 이후에 모델 학습을 진행하면 됩니다.
