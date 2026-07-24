import torch
from torch.utils.data import Dataset

# Header estándar inyectado porque el dataset original carece de cabeceras X:
ABC_HEADER_SINTETICO = "X:1\nT:Generated\nM:4/4\nL:1/8\nK:C\n"

def add_synthetic_header(abc_body: str, header: str = ABC_HEADER_SINTETICO) -> str:
    """Añade un encabezado ABC válido si el texto carece de él para análisis en music21."""
    return header + abc_body.strip()

class ABCDataset(Dataset):
    def __init__(self, text, char2idx, seq_length=100):
        self.seq_length = seq_length
        self.data = [char2idx.get(c, 0) for c in text]
        
    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1], dtype=torch.long)
        return x, y