import torch
from torch.utils.data import Dataset
import json
import re

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

def save_vocab(chars, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False, indent=2)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chars = data["chars"]
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char

def clean_corpus(raw_text: str) -> str:
    def contiene_html(texto: str) -> bool:
        patrones_html = [
            r"<a\s+href=", r"</a>", r"<li>", r"</li>", r"<ul>", r"</ul>",
            r"<body", r"</body>", r"<html", r"</html>", r"<center>", r"</center>"
        ]
        return any(re.search(p, texto, re.IGNORECASE) for p in patrones_html)

    tunes = [t.strip() for t in raw_text.split("\n\n") if t.strip()]
    tunes_limpios = [t for t in tunes if not contiene_html(t)]
    return "\n\n".join(tunes_limpios)