import os
import argparse
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models import LSTMModel, TransformerModel
from utils.dataset import ABCDataset

def main():
    parser = argparse.ArgumentParser(description="Script de Entrenamiento para Modelos Generativos ABC")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], required=True, 
                        help="Arquitectura a entrenar (lstm o transformer)")
    parser.add_argument("--data_path", type=str, default="data/piano-musics-abc-notation.txt", 
                        help="Ruta al archivo de texto con el dataset ABC")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje (learning rate)")
    parser.add_argument("--seq_len", type=int, default=100, help="Tamaño de la ventana de contexto")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Carpeta para guardar los .pt")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Entrenando {args.model.upper()} en dispositivo: {device}")
    print("-> cuDNN deshabilitado para máxima estabilidad en hardware moderno.")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {args.data_path}")

    with open(args.data_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    print(f"-> Vocabulario cargado: {vocab_size} caracteres únicos | Longitud texto: {len(raw_text):,} chars")

    full_dataset = ABCDataset(raw_text, char2idx, seq_length=args.seq_len)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == "lstm":
        model = LSTMModel(vocab_size).to(device)
    else:
        model = TransformerModel(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n--- Iniciando entrenamiento por {args.epochs} épocas ---")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)
        epoch_time = time.time() - start_time

        print(f"Época [{epoch+1:02d}/{args.epochs}] | Tiempo: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, f"{args.model.upper()}_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"   🌟 Nuevo óptimo guardado en: {best_path}")

    last_path = os.path.join(args.save_dir, f"{args.model.upper()}_last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"\n✅ Entrenamiento finalizado. Último modelo en: {last_path}")

if __name__ == "__main__":
    main()