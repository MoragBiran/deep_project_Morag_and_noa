import random
import argparse
from tqdm import tqdm
import numpy as np
import os
import time
import re
import datetime
from sklearn.metrics import accuracy_score
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import FCNet
from loader_models import predata, MyopiaDataset

#Definition of a presser - a global variable that contains arguments you can use
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--data_name', default='cleaned_data.csv', metavar='DIR',
                    help='name of the data')
parser.add_argument('--split_r', type=float, default=0.9, metavar='N',
                    help='Split the data, for example: 90% for training, 10% for test and val')

# Hyperparameter
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--lr', type=int, default=0.0001, metavar='N',
                    help='define the learning rate (default: 0.0001)')

# for Models
parser.add_argument('--model_name', type=str, default='FC', metavar='N',
                    help='The name of the chosen model')
parser.add_argument('--patience', type=int, default=3, metavar='N',
                    help='[atience for scheduler]')
parser.add_argument('--factor', type=float, default=0.8, metavar='N',
                    help='factor for scheduler')
parser.add_argument('--initial_weight_decay', type=float, default=0.1, metavar='N',
                    help='initial_weight_decay for optimizer')
parser.add_argument('--beta', type=float, default=0.98, metavar='N',
                    help='beta for optimizer')
parser.add_argument('--alpha', type=float, default=0.91, metavar='N',
                    help='alpha for optimizer')
#In order for the weights to be fixed and not random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#Used for validation
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.BCEWithLogitsLoss()  # Assuming you used BCEWithLogitsLoss for training
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
#Check that we are not over fitting
    with torch.no_grad():
        for input_data, targets in test_loader:
            input_data, targets = input_data.to(device), targets.to(device)
            prediction = model(input_data)
            loss = criterion(prediction, targets)
            total_loss += loss.item()

            # Calculate the number of correct predictions
            predictions = (torch.sigmoid(prediction) > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()

            total_samples += targets.size(0)

    # Calculate metrics
    average_loss = total_loss / len(test_loader)
    accuracy = (correct_predictions / total_samples) * 100.0

    return average_loss, accuracy

def save_net(path, state):
    tt = str(time.asctime())
    img_name_save = 'net' + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save = img_name_save.replace(' ', '_') + '.pt'
    _dir = os.path.abspath('../')
    path = os.path.join(_dir, path)
    t = datetime.datetime.now()
    datat = t.strftime('%m/%d/%Y').replace('/', '_')
    dir = os.path.join(path, 'net' + '_' + datat)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            print("Directory '%s' created successfully" % ('net' + '_' + datat))
        except OSError as error:
            print("Directory '%s' can not be created" % ('net' + '_' + datat))

    net_path = os.path.join(dir, img_name_save)
    print(net_path)
    torch.save(state, net_path)
    return net_path

def main(args_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_df, test_df, class_weights, mean, std = predata(args_config)


    # Example for creating a correlation heatmap for numeric features
    correlation_matrix = train_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Plot the class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_myopia', data=train_df)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

    train_dataset = MyopiaDataset(train_df, mean, std)
    test_dataset = MyopiaDataset(test_df, mean, std)
    #Gives a different weight to solve the problem of the unbalance in the data from the torch library
    sampler = WeightedRandomSampler(class_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args_config.batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=args_config.batch_size)
    # Define the model & optimizer and loss function
    model = FCNet().to(device)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args_config.lr,
        weight_decay=args_config.initial_weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args_config.factor,
                                  patience=args_config.patience, verbose=True)



    best_loss = float('inf')  # Initialize with a high value
    best_accuracy = 0.0  # Initialize with a low value
    early_stopping_counter = 0
    max_early_stopping = 5
    training_losses = []
    testing_losses = []
    validation_losses = []

    for epoch in range(args_config.epochs):
        model.train()
        train_batch = []
        pbar = tqdm(train_loader, total=len(train_loader))

        for i, (input_data, targets) in enumerate(pbar):
            input_data, targets = input_data.to(device), targets.to(device)
            optimizer.zero_grad()
            prediction = model(input_data)
            loss = criterion(prediction, targets)
            train_batch.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            pbar.set_postfix({'Epoch': epoch,
                              'Training Loss': np.mean(train_batch)
                              })
        train_loss = np.mean(train_batch)

        # Calculate validation loss for this epoch using evaluate_model
        validation_loss, _ = evaluate_model(model, test_loader, device)

        # Append the validation loss for this epoch
        validation_losses.append(validation_loss)



        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        if test_loss < best_loss:
            best_loss = test_loss
            best_accuracy = test_accuracy
            path = os.getcwd()
            saved_model_path = save_net(path, model.state_dict())
            print(f"Saved model to {saved_model_path}")
        # #Test Loss plot
        # testing_losses.append(np.mean(test_loss))
        # plt.figure(figsize=(10, 5))
        # epochs = range(1, len(testing_losses) + 1)
        # plt.plot(epochs, testing_losses, label="Testing Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Testing Loss")
        # plt.legend()
        # plt.title("Testing Loss Over Epochs")
        # plt.grid(True)
        # plt.show()

        # Check for early stopping
        if test_loss > best_loss:
            early_stopping_counter += 1
            if early_stopping_counter >= max_early_stopping:
                print("Early stopping triggered.")
                break

        else:
            early_stopping_counter = 0

        scheduler.step(test_loss)
        #Plot of the epochs
        training_losses.append(np.mean(train_batch))


        #plot Training
        plt.figure(figsize=(10, 5))
        plt.plot(training_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Over Epochs")
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(validation_losses) + 1)

    # Plot validation loss
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.title("Validation Loss Over Epochs")
    plt.grid(True)
    plt.show()



    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')


if __name__ == '__main__':
    args_config = parser.parse_args()
    main(args_config)

