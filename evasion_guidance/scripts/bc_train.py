import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
import torch.optim as optim
import uuid
import yaml
from tqdm import tqdm

from evasion_guidance.scripts.cvae_utils import CVAE, Encoder, Decoder
from evasion_guidance.dataset.evasion_dataset import EvasionDataset


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        name=config["run_name"]
    )
    wandb.run.save()

class NatureCNN(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        n_input_channels = n_input_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 1, 100, 100).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class CVAEModel(nn.Module):
    def __init__(self, hidden_dim, cvae_latent_dim):
        super(CVAEModel, self).__init__()        
        print("Creating model")
        self.heading_size = 2

        self.cnn = NatureCNN()

        self.latent_input_size = self.cnn.features_dim + self.heading_size 
        self.output_size = 20
        self.cvae_latent_dim = cvae_latent_dim

        encoder = Encoder(self.latent_input_size + self.output_size, hidden_dim, cvae_latent_dim)
        decoder = Decoder(cvae_latent_dim + self.latent_input_size, hidden_dim, self.output_size)
        self.model = CVAE(encoder, decoder)

        # self.risk_prediction = nn.Sequential(
        #     nn.Linear(self.latent_input_size + self.output_size, hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )
        
                    
    def forward(self, heat_map, heading, expert_path):        
        img_feature = self.cnn(heat_map)
        combined_input = torch.cat((img_feature, heading), dim=1)
        expert_path_flatten = expert_path.view(expert_path.shape[0], -1)
        mu, logvar, recon_batch = self.model(expert_path_flatten, combined_input)
        
        return mu, logvar, recon_batch, None
    
        # encoded_input = torch.cat((expert_path_flatten, combined_input), 1)
        # predicted_risk = self.risk_prediction(encoded_input)
        # return mu, logvar, recon_batch, predicted_risk

    
    def act(self, heat_map, heading, deterministic=False):
        if deterministic:
            z = torch.zeros(1, self.cvae_latent_dim).to(self.device)
        else:
            z = torch.randn(1, self.cvae_latent_dim).to(self.device)
        img_feature = self.cnn(heat_map)
        combined_input = torch.cat((z, img_feature, heading), dim=1)
        return self.model.decoder(combined_input)

def risk_criteria(init_batch, recon_batch):
    loss_function = torch.nn.MSELoss()
    loss = loss_function(torch.sum(init_batch, dim=1).unsqueeze(dim=1), recon_batch)

    return loss

def criteria(recon_batch, init_batch):
    """ Compute loss function for position and orientation. """
    loss_function = torch.nn.MSELoss()
    recon_batch = recon_batch.view(-1, recon_batch.shape[-1])
    init_batch = init_batch.view(-1, init_batch.shape[-1])
    loss_xy = loss_function(recon_batch[:, 0:2], init_batch[:, 0:2])

    return loss_xy

def kl_regularizer(mu, logvar, kl_weight):
    """ KL Divergence regularizer """
    # it still returns a vector with dim: (batchsize,)
    return kl_weight * 2 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1)

def train(config):
    device = config['device']

    # Create dataset
    data_config = config['data']
    num_data = data_config['num_data']
    params_path = data_config['params_path']
    mcmc_data_path = data_config['mcmc_data_path']
    rrt_data_path = data_config['rrt_data_path']
    dataset = EvasionDataset(mcmc_data_path, rrt_data_path, params_path, num_data, device=device)

    features, labels = dataset[0]

    # Split dataset into training and validation sets
    train_size = int(data_config['training_data_ratio'] * len(dataset))
    val_size = dataset.num_data - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Instantiate the model, loss function, and optimizer
    model_config = config['model']
    hidden_dim = model_config['hidden_dim']
    cvae_latent_dim = model_config['cvae_latent_dim']
    model = CVAEModel(hidden_dim, cvae_latent_dim).to(device)
    model.device = device

    # Create DataLoaders
    train_config = config['train']
    batch_size = train_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_epochs = train_config['num_epochs']
    lr = train_config['lr']
    kl_weight = train_config['kl_weight']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Set up the learning rate scheduler with linear decay
    def lambda_lr(epoch):
        return 1 - epoch / num_epochs

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Set up wandb writer
    experiment_id = str(uuid.uuid4())[:8]
    run_name = config['name'] + '_cvae_'  + experiment_id
    print("Run Name: ", run_name)

    # Directory to save checkpoints
    checkpoint_dir = 'checkpoints/' + run_name
    os.makedirs(checkpoint_dir, exist_ok=False)

    # Save the config for testing
    config['run_name'] = run_name

    with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # wandb_init(config)


    # Lists to store loss values
    train_losses = []
    val_losses = []

    # Initialize variables to track the best model
    best_val_loss = float('inf')
    best_epoch = 0


    print(f"Training started on {device}")
    global_steps = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')):
            heat_map = features['heat_map'].to(device)
            heading = features['heading'].to(device)
            reference_path = labels['desired_path'].to(device)
            # reference_path_risks = labels['desired_path_risk'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            mu, logvar, recon_batch, _ = model(heat_map, heading, reference_path)
            recon_loss = criteria(recon_batch, reference_path.view(reference_path.shape[0], -1))

            kl_loss = kl_regularizer(mu, logvar, kl_weight)
            kl_loss_mean = torch.mean(kl_loss, 0)
            loss = 100.0*recon_loss + kl_loss_mean
            
            ### Uncomment if predicting risks. ###
            # recon_risk_loss = risk_criteria(reference_path_risks, recon_batch_risks)
            # loss = 1000.0*recon_loss + 10.0*kl_loss_mean + 100.0*recon_risk_loss

            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if global_steps % 1000 == 0:
            #     wandb.log({'train_loss': loss.item(),
            #                "train_recon_loss": recon_loss.item(),
            #                'train_kl_loss': kl_loss_mean.item()}, step=global_steps)
            
            # global_steps += 1
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        # Step the learning rate scheduler
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # print("Validation...")
            for _, (features, labels) in enumerate(val_loader):
                heat_map = features['heat_map'].to(device)
                goal_direction = features['heading'].to(device)
                reference_path = labels['desired_path'].to(device)
                # reference_path_risks = labels['desired_path_risk'].to(device)


                # Forward pass
                mu, logvar, recon_batch, _ = model(heat_map, goal_direction, reference_path)
                recon_loss = criteria(recon_batch, reference_path.view(reference_path.shape[0], -1))

                kl_loss = kl_regularizer(mu, logvar, kl_weight)
                kl_loss_mean = torch.mean(kl_loss, 0)

                loss = 1000.0*recon_loss + kl_loss_mean

                ### Uncomment if predicting risks. ###
                # recon_risk_loss = risk_criteria(reference_path_risks, recon_batch_risks)
                # loss = 1000.0*recon_loss + 10.0*kl_loss_mean + 100.0*recon_risk_loss

                val_loss += loss.item()

                # wandb.log({'validation_loss': loss.item(),
                #            "validation_recon_loss": recon_loss.item(),
                #            'validation_kl_loss': kl_loss_mean.item()}, step=global_steps)


        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}')

        # Save the model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)



    print("Run Name: ", run_name)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a policy with BC.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    with open(args.config,"r") as file_object:
        config = yaml.load(file_object,Loader=yaml.SafeLoader)

    train(config)
