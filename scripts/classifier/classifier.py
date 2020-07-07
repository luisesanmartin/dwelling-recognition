import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import net
from loader import dwellingsDataset

def loss_nll(predicts, targets):
    '''
    '''

    loss = nn.NLLLoss()
    output = loss(predicts, targets)

    return output

def train(dataset, model, optimizer, epoch=None, model_name='../../models/kampala_classifier.pkl'):
    '''
    '''

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device=device)
    model.train()

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

        if epoch and idx % 20 == 0:
            print('Epoch # %d, Loss: %.4f' (epoch+1, loss.item()))

    # Saving the model
    torch.save(model, model_name)

    # Evaluating accuracy
    model.eval()
    output = model(x)
    predictions = torch.argmax(output.data, dim=1).numpy()
    num_correct = (predictions == y.numpy()).sum()
    num_samples = y.size(0)
    print('\nEpoch #', epoch)
    print('Accuracy:', round(num_correct/num_samples*100, 2), '%')
    print('\n')

def main():

    classifier = net().float()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    dataset = dwellingsDataset()
    data = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    epochs = 10

    for epoch in range(epochs):
        train(data, classifier, optimizer, epoch)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    main()
