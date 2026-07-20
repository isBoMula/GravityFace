import pickle
from PIL import Image
from torch.utils.data import Dataset

class my_ImageFolder(Dataset):
    def __init__(self, img_list_path, transform=None, target_transform=None):
        file = open(img_list_path, "rb")
        self.img_list = pickle.load(file)
        file.close()
        
        self.length = len(self.img_list)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, target = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

if __name__ == "__main__":
    train_set = my_ImageFolder("ms1mv3.pickle")