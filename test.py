import pickle
import numpy as np
from tqdm import tqdm

# def load_data():
#     print("Loading data...")
#     dataset_X,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))

#     print(dataset_X)
#     print(Y)

#     print("Data loaded!")

#     return np.asarray(dataset_X), np.asarray(Y)

# ds,y = load_data()

# print(y.shape)
# print(np.unique(y,return_counts=True))


x_train = [[0,0],[1,0],[0,0]]
unique, count = np.unique(x_train, return_counts=True, axis = 0)
print(len(count))