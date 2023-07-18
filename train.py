
import torch
from model import MLPModel
from dataloader import load_processed_data, load_data, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
import tqdm
#ds = load_processed_data()
model = MLPModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

from torch.utils.data import DataLoader
epochs = 100
batch_size = 64

training_data = GeneticDataset()
train_length = len(training_data)

test_data = GeneticDataset(train=False)
scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=int(train_length/batch_size*epochs))
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

#check if test and train data have overlapp
# for i, data in enumerate(test_dataloader):
#     inputs,_ = data
#     print("at batch {}".format(i))
#     for j, data2 in enumerate(train_dataloader):
#         inputs2,_ = data2
#         #print("at pos {}".format(j))
#         if torch.equal(inputs, inputs2):
#             print("overlap detected")
#             break

loss_fn = torch.nn.CrossEntropyLoss()
running_loss = 0.0
wandb.init(project="disease_prediction")
best_acc = 0.0
for epoch in range(epochs):
    for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)
        #  print(outputs.shape)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            l1_lambda = 1e-4
            l1_norm = sum([torch.norm(p, p=2) for p in model.parameters()])
            loss += l1_lambda * l1_norm
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            scheduler.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1 == 0:
                #print accuracy of current batch
                
                acc = np.sum(np.argmax(outputs.detach().numpy(), axis = 1) == labels.detach().numpy())/labels.shape[0]

           #     print("accuracy: {}".format(acc))
                last_loss = running_loss / 1 # loss per batch
                log_dict = {"loss": last_loss, "accuracy": acc, "lr": scheduler.get_lr()[0]}
                wandb.log(log_dict)
                if i % 10 == 0:
                    print('  batch {} loss: {} accuracy: {}'.format(i + 1, last_loss, acc))
                running_loss = 0.
    #evaluate model
    with torch.no_grad():
        accummulated_acc = 0.0
        accummulated_loss = 0.0
        for i, data in enumerate(test_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            acc = np.sum(np.argmax(outputs.detach().numpy(), axis = 1) == labels.detach().numpy())/labels.shape[0]
            accummulated_acc += acc
            accummulated_loss += loss.item()
        wandb.log({"test_loss": accummulated_loss/len(test_dataloader), "test_accuracy": accummulated_acc/len(test_dataloader)})
        print(' test epoch {} loss: {} accuracy: {}'.format(epoch+1, accummulated_loss/len(test_dataloader), accummulated_acc/len(test_dataloader)))
    if accummulated_acc/len(test_dataloader) > best_acc:
        best_acc = accummulated_acc/len(test_dataloader)
        torch.save(model.state_dict(), "models/model"+ str(best_acc) +".pt")
  #  model.save("model"+str(epoch)+".pt")