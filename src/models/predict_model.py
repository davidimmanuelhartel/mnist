import os
import pathlib
from typing import Dict, List, Tuple
import wandb
import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.pyplot import show
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, train:bool=True):
        
        # 3. Create class attributes
        self.classes = list(range(0, 10))
        self.class_to_idx = {str(idx):idx for idx in self.classes}

        # Import preprocessed data
        train_test_data = torch.load(os.path.join(targ_dir, 'train_test_processed.pt'))
        if train:
            self.images = train_test_data['train_data']
            self.labels = train_test_data['train_labels']
        else:
            self.images = train_test_data['test_data']
            self.labels = train_test_data['test_labels']

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:

        return self.images.shape[0]
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.images[index]
        class_name  = str(int(self.labels[index]))
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        return img, class_idx # return data, label (X, y)

@click.command()
@click.option("--checkpoint", default=os.path.join('src', 'models', 'my_trained_model.pt'), help='Path to file with state dict of the model')
def evaluate(checkpoint):

    print("Predicting day and night")

    # TODO: Implement evaluating loop here
    model = MyAwesomeModel()
    
    testset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=False)
    # Test DataLoader
    testloader = DataLoader(dataset=testset, # use custom created train Dataset
                                        batch_size=64, # how many samples per batch?
                                        shuffle=True) # shuffle the data?
    
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    # logging with wandb

    wandb.init(project = "MNIST Evaluation")
    wandb.watch(model, log_freq=100)

    test_accuracies = []

    with torch.no_grad():
         # Set model to evaluation mode to turn off dropout
        model.eval()
        for images, labels in testloader:
            # Flatten images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            # Caculate log probabilities
            log_ps = model(images)
            # Calculate probabilities
            ps = torch.exp(log_ps)
            # Obtain top probabilities and corresponding classes
            top_p, top_class = ps.topk(1, dim=1)
            # Compare with true labels
            equals = top_class == labels.view(*top_class.shape)   
            # Obtain accuracy
            batch_acccuracy = torch.mean(equals.type(torch.FloatTensor))       
            test_accuracies.append(batch_acccuracy) 


    accuracy = sum(test_accuracies)/len(testloader)  

    plt.imshow(  images[1].permute(1, 2, 0)  )

    # print(f'Accuracy on test: {accuracy.item()*100}%')
    
    # my_table = wandb.Table()
    # images_t = [wandb.Image(im) for im in images]
    # print(len(images_t),len(labels.numpy()),len(top_class.numpy()))
    # my_table.add_column("image", images_t)
    # my_table.add_column("label", labels.numpy())
    # my_table.add_column("class_prediction", top_class.numpy())

    # # Log your Table to W&B
    # wandb.log({"mnist_predictions": my_table})

    
if __name__ == "__main__":
    evaluate()
