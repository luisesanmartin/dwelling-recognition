import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from nets import net
from loader import dwellingsDataset

def loss_nll(predicts, targets):
    '''
    '''

    loss = nn.NLLLoss()
    output = loss(predicts, targets)

    return output

def train(dataset, model, optimizer, epoch=None, loss_file=None, accuracy_file=None):
    '''
    '''

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device=device)
    model.train()

    # Training
    for idx, (x, y) in enumerate(dataset):
        x = x.float()
        x = x.to(device=device)
        y = y.long()
        y = y.to(device=device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_nll(output, y)
        loss.backward()
        optimizer.step()

        # Saving the loss in the results file:
        if epoch and loss_file:
            with open(loss_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([int(epoch), int(idx), loss.item()])

        if epoch and idx % 20 == 0:
            print("Epoch %d, Loss: %.4f" % (epoch, loss.item()))

    # Saving the model
    if epoch:
        torch.save(model, results_path+'kampala_classifier.pkl')

    # Evaluating accuracy
    model.eval()
    output = model(x)
    predictions = torch.argmax(output.data, dim=1).cpu().numpy()
    num_correct = (predictions == y.cpu().numpy()).sum()
    num_samples = y.size(0) * y.size(1) * y.size(2)
    accuracy = num_correct/num_samples*100
    if epoch:
        print('\nEpoch #', epoch)
    print('Accuracy:', round(accuracy, 2), '%')
    print('\n')
    if epoch and accuracy_file:
        with open(accuracy_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([int(epoch), accuracy])

def main():

    classifier = net().float()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    dataset = dwellingsDataset()
    data = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    epochs = 2
    results_path = '../../models/'

    # Deleting csv files in results_path:
    loss_file = results_path + 'kampala_classifier_losses.csv'
    accuracy_file = results_path + 'kampala_classifier_accuracy.csv'
    try:
        os.remove(loss_file)
    except OSError:
        print('File', loss_file, 'does not exist yet')
    try:
        os.remove(accuracy_file)
    except OSError:
        print('File', accuracy_file, 'does not exist yet')
    
    # Creating new csv files with results of classifier:
    with open(loss_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Iteration', 'Loss'])
    with open(accuracy_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy'])

    # Training the net
    for epoch in range(epochs):
        train(data, classifier, optimizer, epoch, loss_file, accuracy_file)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    main()
