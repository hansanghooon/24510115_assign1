import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class MNIST(Dataset):
    """ MNIST dataset class for loading and preprocessing the MNIST dataset images.

    Args:
        data_dir: directory path containing images
        train: boolean indicating if the subset is training data
    """
    def __init__(self, data_dir, train=True):
        # train 이면 하위파일 train, 아니면 test 
        self.data_dir = os.path.join(data_dir, 'train' if train else 'test')
        
        # Store all image filenames
        self.filenames = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        
        # Define the transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors with values in [0, 1]
            transforms.Normalize(mean=[0.1307], std=[0.3081])  # Normalize the images
        ])

    def __len__(self):
        # Return the number of files in the dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the filename of the idx-th image
        filename = self.filenames[idx]
        
        # Load the image
        img = Image.open(os.path.join(self.data_dir, filename)).convert('L')  # Convert to grayscale
        
        # Apply transformations
        img = self.transform(img)
        
        # Extract label from the filename
        label = int(filename.split('_')[1].split('.')[0])
        
        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

if __name__ == '__main__':
    # Define the paths to the data directories
    base_dir = r'C:\deep\mnist-classification\data'
    
    # Create dataset instances for training and testing
    train_dataset = MNIST(data_dir=base_dir, train=True)
    test_dataset = MNIST(data_dir=base_dir, train=False)
    
    # Print dataset information to verify
    print('Training dataset size:', len(train_dataset))
    print('Testing dataset size:', len(test_dataset))
    train_img, train_label = train_dataset[0]  # Get the first training sample
    test_img, test_label = test_dataset[0]  # Get the first testing sample
    
    print('First training image shape:', train_img.shape, 'Label:', train_label)
    print('First testing image shape:', test_img.shape, 'Label:', test_label)
