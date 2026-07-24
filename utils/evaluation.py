import torch
import torch.nn.functional as F
from music21 import converter
from tqdm import tqdm
from .dataset import add_synthetic_header

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_valid_abc_silent(abc_text, use_synthetic_header=False):
    text_to_eval = abc_text
    if use_synthetic_header and not abc_text.strip().startswith("X:"):
        text_to_eval = add_synthetic_header(abc_text)
        
    try:
        converter.parse(text_to_eval, format="abc")
        return True
    except Exception:
        return False

@torch.no_grad()
def generate_sample(model, seed_text, char2idx, idx2char, device, length=300, temperature=0.8, context_window=100):
    model.eval()
    chars = [char2idx.get(c, 0) for c in seed_text]
    input_seq = torch.tensor([chars], dtype=torch.long, device=device)
    hidden = None
    generated = seed_text

    for _ in range(length):
        logits, hidden = model(input_seq, hidden)
        last_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_logits, dim=0)
        
        next_idx = torch.multinomial(probs, num_samples=1).item()
        generated += idx2char[next_idx]
        
        if hidden is not None:
            input_seq = torch.tensor([[next_idx]], dtype=torch.long, device=device)
        else:
            recent_chars = [char2idx.get(c, 0) for c in generated[-context_window:]]
            input_seq = torch.tensor([recent_chars], dtype=torch.long, device=device)

    return generated

@torch.no_grad()
def evaluate_syntactic_validity(model, seed_text, char2idx, idx2char, device, n_samples=100, length=300, temperature=0.8, use_synthetic_header=False, model_name="Modelo"):
    model.eval()
    valid_count = 0
    
    print(f"\n--- Evaluando Validez Sintáctica: {model_name} ({n_samples} muestras | T={temperature}) ---")
    for _ in tqdm(range(n_samples), desc=f"Muestreando ({model_name})"):
        song = generate_sample(model, seed_text, char2idx, idx2char, device, length, temperature)
        if is_valid_abc_silent(song, use_synthetic_header=use_synthetic_header):
            valid_count += 1
            
    validity_rate = (valid_count / n_samples) * 100.0
    print(f"-> {model_name}: {valid_count}/{n_samples} composiciones válidas ({validity_rate:.2f}%)\n")
    return validity_rate