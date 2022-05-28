import torch

from cigin.model import CIGINModel
from cigin.dataset import data_to_loader


class CIGINPredictor:

    def __init__(self, model_path="weights/best_model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CIGINModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, solute_smiles: str, solvent_smiles: str):
        data = data_to_loader(solute_smiles, solvent_smiles)
        delta_g, interaction_map = self.model(
            data[0].to(self.device),
            data[1].to(self.device),
            torch.tensor(data[2]).to(self.device),
            torch.tensor(data[3]).to(self.device),
        )
        interaction_map = interaction_map.cpu().detach().numpy()
        delta_g = delta_g.cpu().detach().item()
        return delta_g
