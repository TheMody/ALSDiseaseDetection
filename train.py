
import torch
from model import Model
from dataloader import load_processed_data, load_data, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split


def checktrainTestOverlap(train_dataloader,test_dataloader):
    for i, data in enumerate(test_dataloader):
        inputs,_ = data
        print("at batch {}".format(i))
        for j, data2 in enumerate(train_dataloader):
            inputs2,_ = data2
            #print("at pos {}".format(j))
            if torch.equal(inputs, inputs2):
                print("overlap detected")
                break


if __name__ == "__main__":
    epochs = 100
    batch_size = 64
    l1_lambda = 1e-3

    #basic building blocks
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="disease_prediction")

    #data
    geneticData = GeneticDataset()

    # Split dataset into train and validation sets
    train_indices, test_indices = train_test_split(list(range(len(geneticData))), test_size=0.2, random_state=42)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
   
    train_dataloader = DataLoader(geneticData, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(geneticData, batch_size=batch_size, sampler=test_sampler)

    scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=int(len(geneticData)/batch_size*epochs))

    running_loss = 0.0
    best_acc = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
                # Every data instance is an input + label pair
                inputs, labels = data

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)
                # print(outputs.shape)
                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                l1_norm = sum([torch.norm(p, p=1) for p in model.parameters()])
                loss += l1_lambda * l1_norm
                loss.backward()

                # Adjust learning weights
                optimizer.step()
                scheduler.step()

                # Gather data and report
                log_freq = 1
                running_loss += loss.item()
                if i % log_freq == 0:
                    acc = np.sum(np.argmax(outputs.detach().numpy(), axis = 1) == labels.detach().numpy())/labels.shape[0]
                    avg_loss = running_loss / log_freq # loss per batch
                    log_dict = {"avg_loss": avg_loss, "accuracy": acc, "lr": scheduler.get_lr()[0]}
                    wandb.log(log_dict)
                    running_loss = 0.
                    # if i % 10 == 0:
                    #     print('  batch {} loss: {} accuracy: {}'.format(i + 1, avg_loss, acc))


        #evaluate model after epoch
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
            avg_acc = accummulated_acc/len(test_dataloader)
            avg_loss = accummulated_loss/len(test_dataloader)
            wandb.log({"test_loss": avg_loss, "test_accuracy": avg_acc})
            print(' test epoch {} loss: {} accuracy: {}'.format(epoch+1, avg_loss, avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "models/model"+ str(best_acc) +".pt")
            print(f"saved new best model with acc {best_acc}")