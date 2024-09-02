## Imports Machine Learning Library
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

#%matplotlib inline
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

## Imports Library for Streamlit
from stqdm import stqdm
import queue
import time
from tqdm import tqdm
import csv
import streamlit as st
import pandas as pd 

## Loading MNIST Data set 
mnist_dataset = MNIST(root = 'data/', train = True, transform = transforms.ToTensor())
train_data, validation_data = random_split(mnist_dataset, [50000, 10000])
input_size = 28 * 28
num_classes = 10

## I use the model from the site: https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch 


## Model that used in our problem 
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return(out)
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        return(loss)
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
    
model = MnistModel()

## Function for evaluating the model 
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return(torch.tensor(torch.sum(preds == labels).item()/ len(preds)))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))

def save_results(lr, num_epochs, total_time_elapsed, final_accuracy):
    with open('training_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([lr, num_epochs, total_time_elapsed, final_accuracy])

## fit function 
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    total_time_start = time.time()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        ## Training Phase
        model.train()
        train_losses = []
        train_loader_tqdm = stqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for batch in train_loader_tqdm:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            train_loader_tqdm.set_postfix({'Training Loss': loss.item()})

        ## Validation phase
        result = evaluate(model, val_loader)
        history.append(result)
        
    total_time_end = time.time()
    total_time_elapsed = total_time_end - total_time_start
    final_accuracy = history[-1]['val_acc']
    st.text(f"Total training time: {total_time_elapsed:.2f} seconds")
    st.text(f"Accuracy: {final_accuracy:.2f}")
    save_results(lr, num_epochs, total_time_elapsed, final_accuracy)
    return history

## Writing to the website 
hyperparameters = queue.Queue()
st.title('MNIST Model Training')

## Enter the parameter 
num_training = st.number_input('Enter the number of training example:', min_value=1, max_value=1024, value=3, step=1)

for i in range(num_training): 
    st.text(f"Enter the parameter for the {i+1}th training")
    learning_rate = st.number_input(f"{i+1}th learning rate:", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")
    batch_size = st.number_input(f"{i+1}th batch size:", min_value=1, max_value=1024, value=64, step=1)
    num_epochs = st.number_input(f"{i+1}th number of epochs:", min_value=1, max_value=100, value=10, step=1)
    hyperparameters.put((learning_rate, batch_size, num_epochs))

if st.button('Start Training'):
    while not hyperparameters.empty(): 
        learning_rate, batch_size, num_epochs = hyperparameters.get()
        train_loader = DataLoader(train_data, batch_size, shuffle = True)
        val_loader = DataLoader(validation_data, batch_size, shuffle = False)
        df = pd.read_csv('training_results.csv')
        temp = True
        for i in range(len(df)):
            if (df['Learning rate'][i] == learning_rate and df['Number of epochs'][i] == num_epochs):
                st.text('This combination of learning rate and number of epochs has already been trained.')
                st.text(f"Total training time: {df['Training time (s)'][i]:.2f} seconds")
                st.text(f"Accuracy: {df['Accuracy'][i]:.2f}")
                temp = False
                break        
        if temp == True:
            st.write(f'Starting training with learning rate: {learning_rate}, batch size: {batch_size}, and number of epochs: {num_epochs}')
            result0 = evaluate(model, val_loader)
            history = [result0]
            history_temp = fit(num_epochs, learning_rate, model, train_loader, val_loader)
            history += history_temp
            accuracies = [result['val_acc'] for result in history]
            plt.plot(accuracies, '-x')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('Accuracy Vs. No. of epochs')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

if st.button("Show History"): 
    st.subheader('Training History:')
    st.write(pd.read_csv('training_results.csv'))