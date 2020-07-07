import sys
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

def train(dataset, model, optimizer, epoch=None, model_name='../../models/kampala_classifier.pkl', loss_df='../../models/kampala_classifier_losses.csv'):
    '''
    '''

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device=device)
    model.train()

    df = pd.DataFrame(columns=['epoch', 'iteration', 'loss'])

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
        size = len(df)
        df.loc[size] = [epoch, idx, loss.item()]

        if epoch and idx % 20 == 0:
            print("Epoch %d, Loss: %.4f" % (epoch+1, loss.item()))

    # Saving the model and  loss df
    torch.save(model, model_name)
    df.to_csv(loss_df, index=False)

    # Evaluating accuracy
    model.eval()
    output = model(x)
    predictions = torch.argmax(output.data, dim=1).cpu().numpy()
    num_correct = (predictions == y.cpu().numpy()).sum()
    num_samples = y.size(0) * y.size(1) * y.size(2)
    print('\nEpoch #', epoch)
    print('Accuracy:', round(num_correct/num_samples*100, 2), '%')
    print('\n')

def main():

    classifier = net().float()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    dataset = dwellingsDataset()
    data = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    epochs = 50

    for epoch in range(epochs):
        train(data, classifier, optimizer, epoch)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    main()
