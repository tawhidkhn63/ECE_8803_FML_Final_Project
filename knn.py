import torch
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import pickle

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}

mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize,
])

class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)

        # Shuffle the dataset and select only 10% of the data
        #self.annot = self.annot.sample(frac=0.1, random_state=42)
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)

    def __getitem__(self, index):
        img, target = Image.open(os.getcwd()+self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self._labels)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type=str, default='df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type=str, default='df_prime_test.csv')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--number_neighbors', type=int, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if os.path.exists('train_data.pkl'):
        with open('train_data.pkl', 'rb') as f:
            X_train, y_train = pickle.load(f)
        with open('test_data.pkl', 'rb') as f:
            X_test, y_test = pickle.load(f)
    else:
	    trainset = OCTDataset(args, 'train', transform=transform)
	    testset = OCTDataset(args, 'test', transform=transform)
	    
	    # Create data loaders to load the data in batches
	    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
	    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

	    # Get the batch of data
	    X_train, y_train = next(iter(trainloader))
	    X_test, y_test = next(iter(testloader))

	    # Convert the PyTorch tensors to numpy arrays
	    X_train, y_train = X_train.numpy(), y_train.numpy()
	    X_test, y_test = X_test.numpy(), y_test.numpy()
	   	# x train and test need to have dimensions of 2 but it has dimensions of 4
	    num_samples, num_channels, height, width = X_train.shape
	    print(num_channels, num_samples)
	    X_train = X_train.reshape(num_samples, num_channels * height * width)
	    num_samples, num_channels, height, width = X_test.shape
	    X_test = X_test.reshape(num_samples, num_channels * height * width)
	    print(num_channels, num_samples)
	    # Pickle the X_train and y_train numpy arrays
	    with open('train_data.pkl', 'wb') as f:
	        pickle.dump((X_train, y_train), f)

	    # Pickle the X_test and y_test numpy arrays
	    with open('test_data.pkl', 'wb') as f:
	        pickle.dump((X_test, y_test), f)

    print("Data split finished")

    # Create a KNN classifier with k=5
    knn = KNeighborsClassifier(n_neighbors=args.number_neighbors)
    print("knn classifier created")


    # Train the classifier on the training set
    knn.fit(X_train, y_train)
    print("knn classifier fitted")

    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    print("predictions generated on test dataset")

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}")

	# Calculate the precision of the classifier
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Precision: {precision:.2f}")

	# Calculate the recall of the classifier
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Recall: {recall:.2f}")

	# Calculate the f1 score of the classifier
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.2f}")

	# Calculate the true positive rate of the classifier
    true_positive_rate = np.sum(y_pred.reshape(-1,1) * y_test.reshape(-1,1))/np.sum(y_test)
    print(f"True Positive Rate: {true_positive_rate:.2f}")

	# Calculate the false positive rate of the classifier
    false_positive_rate = np.sum(y_pred.reshape(-1,1) * (1 - y_test).reshape(-1,1))/np.sum(1 - y_test)
    print(f"False Positive Rate: {false_positive_rate:.2f}")




