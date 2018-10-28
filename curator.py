#!/usr/bin/env python3

import os
import numpy as np
from itertools import combinations
from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

def clear():
    """Clears jupyter cell output"""
    try:
        clear_output()
    except:
        pass

class Numpize(nn.Module):
    """ Converts our output into a numpy array"""
    def __init__(self):
        super(Numpize, self).__init__()
        
    def forward(self, x):
        return x.cpu().detach().numpy().reshape(x.shape[0],-1)

class ImageDataset(Dataset):
    """ Standard pytorch dataset"""
    def __init__(self, directory, device, img_size=224):
        self.loader = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        self.imgs = os.listdir(directory)
        self.directory = directory
        self.device = device 
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = os.path.join(self.directory, self.imgs[idx])
        img = Image.open(name).convert('RGB')
        return self.loader(img).to(self.device, torch.float)
    
class Curator:
    
    def __init__(self, path):
        
        # gpu or cpu?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # get our dataset, resize images to 224
        self.dataset = ImageDataset(path, self.device, 224)
        self.n_images = len(self.dataset)
        
        # loader will load them in 16 at a time
        loader = DataLoader(self.dataset, batch_size=16, shuffle=False)

        # I'm gonna use the first few layers from vgg19 to get content simularity
        vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # build a model from the first few layers of the vgg19
        model = nn.Sequential()
        i = 0
        j = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            else:
                name = f'thing_{j}'
                j += 1
            model.add_module(name, layer)
            
            if name == 'conv_4':
                # this is the last layer we want want from vgg
                # add some pooling to get a smaller numpy vector
                model.add_module('pool',nn.AdaptiveAvgPool2d(1))
                model.add_module('numpy', Numpize())
                break

        # store our image vectors here
        img_vecs = np.zeros((self.n_images, 128), dtype=np.float)
        count = 0
        for batch in loader:
            # fetch vectors and store
            vecs = model(batch)
            img_vecs[count:count+vecs.shape[0]] = vecs
            count += vecs.shape[0]

        # get the mean squared error between all pairs of image vectors
        self.pairs = list(combinations(range(self.n_images), 2))
        mse = np.mean((img_vecs[self.pairs,:][:,0,:] - img_vecs[self.pairs,:][:,1,:])**2,1)
        self.results = sorted(list(zip(self.pairs,mse)), key=lambda x: x[1])
        
        # this is list of files we want to purge
        self.remove = []

    def duplicate_detection(self):
    
        # go through pairs of images in order of most similar
        for pair,mse in self.results:
            img0, img1 = pair
            
            # if they've already been marked for removal we can skip
            if img0 in self.remove or img1 in self.remove:
                continue
            
            img0 = Image.open(os.path.join(self.dataset.directory,self.dataset.imgs[img0])).convert('RGB')
            img1 = Image.open(os.path.join(self.dataset.directory,self.dataset.imgs[img1])).convert('RGB')
        
            plt.subplot(121)
            plt.imshow(img0)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(122)
            plt.imshow(img1)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            menu = "1. Add left image to purge list\n2. Add right image to purge list\n3. Add both images to purge list\n4. Skip\n5. Done with dup detection\n"
            choice = str(input(menu))
            if choice == '1':
                self.remove.append(pair[0])
            elif choice == '2':
                self.remove.append(pair[1])
            elif choice == '3':
                self.remove = self.remove + list(pair)
            elif choice == '5':
                break
            else:
                clear()
                continue
            clear()
            
    def garbage_detection(self):    
    
        # garbage images are probably the most disimilar in the dataset
        scores = np.zeros(self.n_images)
        for pair,mse in self.results:
            scores[pair[0]] += mse
            scores[pair[1]] += mse
        
        # get index in reverse order (most similar last)
        for idx in scores.argsort()[::-1]:
            if idx in self.remove:
                continue
            
            img = Image.open(os.path.join(self.dataset.directory,self.dataset.imgs[idx])).convert('RGB')
        
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            menu = "1. Add image to purge list\n2. Skip\n3. Done with garbage detection\n"
            choice = str(input(menu))
            if choice == '1':
                self.remove.append(idx)
            elif choice == '3':
                break
            else:
                clear()
                continue
            clear()

    def purge(self):
        # remove the images from the directory if they were added to purge list
        for img in self.remove:
            file_name = self.dataset.imgs[img]
            os.remove(os.path.join(self.dataset.directory,file_name))
        self.remove = []




