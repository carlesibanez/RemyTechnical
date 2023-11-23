import torch
from utils.PansEggsDataset import PansEggsDataset 
from models.unet import UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=1)
    # parser.add_argument('--data-path', type=str, default='../dataset/train')
    # parser.add_argument('--learning-rate', type=float, default=0.05)
    # parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    FLAGS = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # Create dataset
    dataset = PansEggsDataset(folder_path=FLAGS.data_dir, masks=True, new_size=(400, 400))
    print('Dataset size: ', len(dataset))

    # Split dataset into train and validation
    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=gen)
    print('Train size: ', len(train_dataset))
    print('Validation size: ', len(val_dataset))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)

    # Create model
    model = UNet(in_channels=3, classes=3).to(device)
    print(model)

    # Create loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    starting_epoch = 0

    # Not in use for sagemaker
    load_model = False #'models/unet_ep99.pth'
    if load_model:
        checkpoint       = torch.load(load_model)
        starting_epoch   = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Model loaded successfully. Starting at epoch: {starting_epoch}.')

    test_name = 'test1'
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(FLAGS.output_dir, 'runs', test_name))   #f'runs/{test_name}')

    # Train the model
    for epoch in range(FLAGS.epochs):
        print('Epoch {}/{}'.format(epoch + 1, FLAGS.epochs))
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
        
        if epoch == 0: 
            writer.add_image('Images', inputs[0]*255.0, epoch)
            writer.add_image('Mask', torch.clamp(labels[0]*128, 0, 255).unsqueeze(0), epoch)
        writer.add_image('Outputs', outputs[0], epoch)
        writer.add_image('Outputs_bw', torch.clamp(outputs[0]*128, 0, 255), epoch)

        # Save model
        if epoch == FLAGS.epochs - 1:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(FLAGS.model_dir, 'model.pth'))    # f'checkpoints/{test_name}/unet_ep{epoch}.pth')
