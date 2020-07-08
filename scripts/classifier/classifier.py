import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from nets import net, net2
from loader import dwellingsDataset

def loss_nll(predicts, targets):
    '''
    '''

    loss = nn.NLLLoss()
    output = loss(predicts, targets)

    return output

def train(dataset, model, optimizer, epoch=None, loss_file=None, accuracy_file=None, net_file=None):
    '''
    '''

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device=device)
    model.train()
    if loss_file:
        losses = []

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
        if loss_file:
            losses.append([int(epoch), int(idx), loss.item()])

        if idx % 20 == 0:
            print("Epoch %d, Loss: %.4f" % (epoch, loss.item()))

    # Saving the model
    if net_file:
        torch.save(model, net_file)
    if loss_file:
        with open(loss_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for loss_row in losses:
                writer.writerow(loss_row)

    # Evaluating accuracy
    model.eval()
    output = model(x)
    predictions = torch.argmax(output.data, dim=1).cpu().numpy()
    num_correct = (predictions == y.cpu().numpy()).sum()
    num_samples = y.size(0) * y.size(1) * y.size(2)
    accuracy = num_correct/num_samples*100
    print('\nEpoch #', epoch)
    print('Accuracy:', round(accuracy, 2), '%')
    print('\n')

    # Saving the accuracy result:
    if accuracy_file:
        with open(accuracy_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([int(epoch), accuracy])

def main():

    classifier1 = net().float()
    optimizer1 = optim.Adam(classifier1.parameters(), lr=1e-4)
    classifier2 = net2().float()
    optimizer2 = optim.Adam(classifier2.parameters(), lr=1e-4)
    dataset = dwellingsDataset()
    data = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    epochs = 100
    results_path = '../../models/'

    # Results files:
    loss_file1 = results_path + 'kampala_classifier_losses_net1.csv'
    accuracy_file1 = results_path + 'kampala_classifier_accuracy_net1.csv'
    net_file1 = results_path + 'kampala_classifier_net1.pkl'
    loss_file2 = results_path + 'kampala_classifier_losses_net2.csv'
    accuracy_file2 = results_path + 'kampala_classifier_accuracy_net2.csv'
    net_file2 = results_path + 'kampala_classifier_net2.pkl'
    
    # Creating new csv files with results of classifier:
    with open(loss_file1, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Iteration', 'Loss'])
    with open(accuracy_file1, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy'])
    with open(loss_file2, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Iteration', 'Loss'])
    with open(accuracy_file2, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy'])

    # Training nets
    for epoch in range(epochs):
        train(data, classifier1, optimizer1, epoch, loss_file1, accuracy_file1, net_file1)
        train(data, classifier2, optimizer2, epoch, loss_file2, accuracy_file2, net_file2)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    main()
