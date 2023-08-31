
import torch
from model import MLPModel, EncoderModel, IndMLPModel
from dataloader import load_processed_data, load_data, GeneticDataset, GeneticDatasetpreprocessed
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    gradient_accumulation_steps = 1
    l1_lambda = 1e-5

    #basic building blocks
    model = MLPModel()
   # model = EncoderModel()
   # model = IndMLPModel()
    # print("compiling model")
    # model = torch.compile(model)
    # print("model compiled")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#torch.optim.Adam(model.parameters(), lr=0.001)#
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="disease_prediction")

    #data
    geneticData =GeneticDataset()# GeneticDatasetpreprocessed()#

    # Split dataset into train and validation sets
    train_indices, test_indices = train_test_split(list(range(len(geneticData))), test_size=0.2, random_state=42)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
   
    train_dataloader = DataLoader(geneticData, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(geneticData, batch_size=batch_size, sampler=test_sampler)

   # checktrainTestOverlap(train_dataloader,test_dataloader)

    scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=int(len(train_dataloader)*epochs))

    running_loss = 0.0
    best_acc = 0.0
    for epoch in range(epochs):
        for i in range(int(len(train_dataloader) / gradient_accumulation_steps)):
                optimizer.zero_grad()
                inputs, labels = next(iter(train_dataloader))
                #put data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                accloss = 0.0
                accacc = 0.0
                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                for micro_step in range(gradient_accumulation_steps):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    with torch.no_grad():
                        accacc += torch.sum(torch.argmax(outputs, axis = 1) == labels)/labels.shape[0]
                        accloss += loss.item()

                    inputs, labels = next(iter(train_dataloader))
                    #put data to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # backward pass, with gradient scaling if training in fp16
                    loss.backward()
                    
                acc = accacc/gradient_accumulation_steps
                # Every data instance is an input + label pair
                # inputs, labels = data
                # #put data to device
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                # # Zero your gradients for every batch!
                

                # # Make predictions for this batch
                # outputs = model(inputs)
                # print(outputs.shape)
                # Compute the loss and its gradients
               # loss = loss_fn(outputs, labels)
                # l1_norm = sum([torch.norm(p, p=1) for p in model.parameters()])
                # loss += l1_lambda * l1_norm
              #  loss.backward()

                # Adjust learning weights
                optimizer.step()
                scheduler.step()

                # Gather data and report
                log_freq = 1
                running_loss += accloss
                if i % log_freq == 0:
                 #   acc = np.sum(np.argmax(outputs.detach().cpu().numpy(), axis = 1) == labels.detach().cpu().numpy())/labels.shape[0]
                    avg_loss = running_loss / log_freq # loss per batch
                    log_dict = {"avg_loss": avg_loss, "accuracy": acc, "lr": scheduler.get_lr()[0]}
                    wandb.log(log_dict)
                    running_loss = 0.
                    if i % 20 == 0:
                        print('  batch {} loss: {} accuracy: {}'.format(i + 1, avg_loss, acc))


        #evaluate model after epoch
        with torch.no_grad():
            accummulated_acc = 0.0
            accummulated_loss = 0.0
            for i, data in enumerate(test_dataloader):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                acc = torch.sum(torch.argmax(outputs, axis = 1) == labels)/labels.shape[0]
                accummulated_acc += acc.item()
                accummulated_loss += loss.item()
            avg_acc = accummulated_acc/len(test_dataloader)
            avg_loss = accummulated_loss/len(test_dataloader)
            wandb.log({"test_loss": avg_loss, "test_accuracy": avg_acc})
            print(' test epoch {} loss: {} accuracy: {}'.format(epoch+1, avg_loss, avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "models/model"+ str(best_acc) +".pt")
            print(f"saved new best model with acc {best_acc}")