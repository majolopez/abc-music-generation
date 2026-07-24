import os
import sys
import argparse
import math
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow running this script directly: python scripts/train.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import LSTMModel, TransformerModel
from utils.dataset import ABCDataset, clean_corpus, save_vocab, load_vocab


def main():
    parser = argparse.ArgumentParser(description="Script de entrenamiento para modelos generativos ABC")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], required=True)
    parser.add_argument("--data_path", type=str, default="data/piano-musics-abc-notation.txt")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json",
                         help="Si existe, se reutiliza este vocabulario en vez de reconstruirlo "
                              "(garantiza consistencia con checkpoints ya entrenados). "
                              "Si no existe, se construye desde el corpus limpio y se guarda aquí.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3, help="3e-3 para LSTM, 3e-4 recomendado para Transformer")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=0, help="3 recomendado para Transformer")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--disable_cudnn", action="store_true",
                         help="Usar solo si tu servidor tiene el bug de versiones de cuDNN en conflicto")
    args = parser.parse_args()

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("-> cuDNN deshabilitado (workaround manual solicitado)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Entrenando {args.model.upper()} en dispositivo: {device}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {args.data_path}")

    with open(args.data_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text_original = f.read()

    # Limpieza: descarta tunes contaminados con HTML (ver notebook, paso 2)
    raw_text = clean_corpus(raw_text_original)
    print(f"-> Corpus limpio: {len(raw_text):,} caracteres "
          f"(original: {len(raw_text_original):,})")

    if os.path.exists(args.vocab_path):
        char2idx, idx2char = load_vocab(args.vocab_path)
        vocab_size = len(char2idx)
        print(f"-> Vocabulario reutilizado desde {args.vocab_path}: {vocab_size} caracteres")
    else:
        chars = sorted(list(set(raw_text)))
        vocab_size = len(chars)
        char2idx = {c: i for i, c in enumerate(chars)}
        idx2char = {i: c for i, c in enumerate(chars)}
        os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True) if os.path.dirname(args.vocab_path) else None
        save_vocab(chars, args.vocab_path)
        print(f"-> Vocabulario construido y guardado en {args.vocab_path}: {vocab_size} caracteres")

    full_dataset = ABCDataset(raw_text, char2idx, seq_length=args.seq_len)
    train_size = int(0.9 * len(full_dataset))  # 90/10, igual que en el notebook
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    if args.model == "lstm":
        model = LSTMModel(vocab_size).to(device)
    else:
        model = TransformerModel(vocab_size, context_window=args.seq_len).to(device)

    print(f"-> Parámetros totales: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_perplexity": [], "lr": [], "time": []}

    print(f"\n--- Iniciando entrenamiento por {args.epochs} épocas ---")
    for epoch in range(args.epochs):
        start_time = time.time()

        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)
        epoch_time = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_perplexity"].append(val_ppl)
        history["lr"].append(current_lr)
        history["time"].append(epoch_time)

        print(f"Época [{epoch+1:02d}/{args.epochs}] | Tiempo: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f}) | "
              f"LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, f"{args.model.upper()}_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"   Nuevo óptimo guardado en: {best_path}")

    last_path = os.path.join(args.save_dir, f"{args.model.upper()}_last.pt")
    torch.save(model.state_dict(), last_path)

    history_path = os.path.join(args.save_dir, f"{args.model.upper()}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nEntrenamiento finalizado. Último modelo: {last_path}")
    print(f"Historial guardado en: {history_path}")


if __name__ == "__main__":
    main()