import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def load_data():
    print("Loading data...")
    dataset_X,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))

   # print(dataset_X)
   # print(Y)

    print("Data loaded!")

    return np.asarray(dataset_X), np.asarray(Y)

class GeneticDataset(Dataset):
    def __init__(self, train = True):
        self.train = train

        self.x,self.y = load_data()
     #   self.x = self.x[:int(0.1*len(self.x))]
     #   self.y = self.y[:int(0.1*len(self.y))]
        #now we shuffle x and y
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]
        self.x_train = self.x[:int(0.8*len(self.x))]
        self.y_train = self.y[:int(0.8*len(self.y))]
        self.x_test = self.x[int(0.8*len(self.x)):]
        self.y_test = self.y[int(0.8*len(self.y)):]

    def __len__(self):
        if self.train:
            return len(self.x_train)
        else:   
            return len(self.x_test)

    def __getitem__(self, idx):
        if self.train:
            genome = self.x_train[idx]
            label = self.y_train[idx]
            return genome, label
        else:
            genome = self.x_test[idx]
            label = self.y_test[idx]
            return genome, label


def preprocess_data():
    ds = load_data()
    names = list(ds.columns.values)
    for i in range(len(names)):
        split  = names[i].split(":")
        names[i] = split[0]+":"+split[1]

    _, countunique = np.unique(names, return_counts=True)

    max_length = np.max(countunique)
    print(max_length)
    tokenized_ds = []
    for i in tqdm(range(len(ds))):
        datapoint = np.asanyarray(ds.iloc[i,:].values)
        tokens = []
        current = 0
        for count in countunique:
            token = np.zeros(max_length)
            token[:count] = datapoint[current:current+count]
            tokens.append(token)
            current += count
        datapoint = np.asarray(tokens)
        tokenized_ds.append(datapoint)
    # break

    tokenized_ds = np.asarray(tokenized_ds)
    print(tokenized_ds.shape)
    fileObject = open("data/processed_ds", 'wb')
    pickle.dump(tokenized_ds,fileObject )
    fileObject.close()

#preprocess_data()

def load_processed_data():
    fileObject = open("data/processed_ds", 'rb')
    ds = pickle.load(fileObject)
    fileObject.close()
    return ds



#ds = load_processed_data()