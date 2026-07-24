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
def generate_sample(model, seed_text, char2idx, idx2char, device, length=300,
                     temperature=0.8, top_k=None, context_window=100):
    model.eval()
    chars = [char2idx.get(c, 0) for c in seed_text]
    generated_ids = list(chars)
    generated = seed_text

    for _ in range(length):
        input_ids = generated_ids[-context_window:]
        input_seq = torch.tensor([input_ids], dtype=torch.long, device=device)

        logits, _ = model(input_seq)
        last_logits = logits[0, -1, :] / temperature

        if top_k is not None:
            top_values, top_indices = torch.topk(last_logits, top_k)
            filtered = torch.full_like(last_logits, float("-inf"))
            filtered[top_indices] = top_values
            last_logits = filtered

        probs = F.softmax(last_logits, dim=0)
        next_idx = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_idx)
        generated += idx2char[next_idx]

    return generated

@torch.no_grad()
def evaluate_syntactic_validity(model, seed_text, char2idx, idx2char, device,
                                  n_samples=100, length=300, temperature=0.8, top_k=None,
                                  use_synthetic_header=False, model_name="Modelo"):
    model.eval()
    valid_count = 0

    print(f"\n--- Evaluando Validez Sintáctica: {model_name} ({n_samples} muestras | T={temperature} | top_k={top_k}) ---")
    for _ in tqdm(range(n_samples), desc=f"Muestreando ({model_name})"):
        song = generate_sample(model, seed_text, char2idx, idx2char, device, length, temperature, top_k)
        if is_valid_abc_silent(song, use_synthetic_header=use_synthetic_header):
            valid_count += 1

    validity_rate = (valid_count / n_samples) * 100.0
    print(f"-> {model_name}: {valid_count}/{n_samples} composiciones válidas ({validity_rate:.2f}%)\n")
    return validity_rate