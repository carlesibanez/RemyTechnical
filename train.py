import torch
from utils.PansEggsDataset import PansEggsDataset 
from models.unet import UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# Create dataset
dataset = PansEggsDataset(folder_path='../dataset/train', masks=True, new_size=(400, 400))
print('Dataset size: ', len(dataset))

# Split dataset into train and validation
gen = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=gen)
print('Train size: ', len(train_dataset))
print('Validation size: ', len(val_dataset))

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# Create model
model = UNet(in_channels=3, classes=3).to(device)
print(model)

# Create loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create tensorboard writer
writer = SummaryWriter(log_dir='runs/unet')

def train():
    epochs = 10
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Train
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print('Train loss: {:.4f}'.format(epoch_loss))
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation
        model.eval()
        running_loss = 0.0
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Print statistics
            running_loss += loss.item()
        epoch_loss = running_loss / len(val_loader)
        print('Validation loss: {:.4f}'.format(epoch_loss))
        writer.add_scalar('Loss/val', epoch_loss, epoch)
        
        if epoch == 0: writer.add_images('Images', inputs, epoch)
        writer.add_images('Outputs', outputs, epoch)

        torch.save(model.state_dict(), f'models/unet_ep{epoch}.pth')

    