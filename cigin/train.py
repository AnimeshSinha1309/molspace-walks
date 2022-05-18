import tqdm.auto as tqdm
import torch
import numpy as np

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(max_epochs, model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = 100
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = []
        with tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}") as tq_loader:
            for samples in tq_loader:
                optimizer.zero_grad()
                outputs, interaction_map = model(
                    samples[0].to(device),
                    samples[1].to(device),
                    samples[2].to(device),
                    samples[3].to(device),
                )
                l1_norm = torch.norm(interaction_map, p=2) * 1e-4
                loss = loss_fn(outputs, torch.tensor(samples[4]).to(device).float()) + l1_norm
                loss.backward()
                optimizer.step()
                loss = loss - l1_norm
                running_loss.append(loss.cpu().detach())
                tq_loader.set_postfix(
                    train_loss=np.mean(np.array(running_loss))
                )
            model.eval()

        with tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch}") as tq_loader:
            val_loss_array, val_mae_array = [], []
            for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in tq_loader:
                outputs, i_map = model(
                    solute_graphs.to(device),
                    solvent_graphs.to(device),
                    solute_lens.to(device),
                    solvent_lens.to(device),
                )
                val_loss = loss_fn(outputs, torch.tensor(labels).to(device).float())
                val_mae = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())
                val_loss_array.append(val_loss.cpu().detach().numpy())
                val_mae_array.append(val_mae.cpu().detach().numpy())
                tq_loader.set_postfix(
                    val_loss=np.mean(np.array(val_loss_array).flatten()),
                    mae_loss=np.mean(np.array(val_mae_array).flatten())
                )

            val_loss = np.mean(np.array(val_loss_array).flatten())

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "weights/best_model.pth")
