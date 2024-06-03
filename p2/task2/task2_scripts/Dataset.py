import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
  
  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
      image = self.images[idx]

      # Apply transformations to the image
      if self.transform:
        image = self.transform(image)

      label = int(self.labels[idx])
      return (image, label)
  
def getAllDataloadersClassification(train_imgs, val_imgs, test_imgs, batch_size=32, num_workers=8):
    train_images = [img[0] for img in train_imgs]
    train_labels = [img[1]-1 for img in train_imgs]
    val_images = [img[0] for img in val_imgs]
    val_labels = [img[1]-1 for img in val_imgs]
    test_images = [img[0] for img in test_imgs]
    test_labels = [img[1]-1 for img in test_imgs]

    # Define transformations
    data_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset objects
    training_data = CustomDataset(train_images, train_labels, transform=data_aug)
    validation_data = CustomDataset(val_images, val_labels, transform=data_aug)
    testing_data = CustomDataset(test_images, test_labels, transform=data_aug)

    # Define data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Show one image
    # plt.imshow(training_data[0][0][0,:,:], cmap='gray')

    return train_dataloader, validation_dataloader, test_dataloader

def getAllDataLoadersRegression(train_imgs, val_imgs, test_imgs, batch_size=32, num_workers=8):
    train_images = [img[0] for img in train_imgs]
    train_labels = [img[1] for img in train_imgs]
    val_images = [img[0] for img in val_imgs]
    val_labels = [img[1] for img in val_imgs]
    test_images = [img[0] for img in test_imgs]
    test_labels = [img[1] for img in test_imgs]

    # Define transformations
    data_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset objects
    training_data = CustomDataset(train_images, train_labels, transform=data_aug)
    validation_data = CustomDataset(val_images, val_labels, transform=data_aug)
    testing_data = CustomDataset(test_images, test_labels, transform=data_aug)

    # Define data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Show one image
    # plt.imshow(training_data[0][0][0,:,:], cmap='gray')

    return train_dataloader, validation_dataloader, test_dataloader