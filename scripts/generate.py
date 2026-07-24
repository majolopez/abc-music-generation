import os
import argparse
import torch
from models import LSTMModel, TransformerModel
from utils.evaluation import generate_sample
from utils.dataset import add_synthetic_header

def main():
    parser = argparse.ArgumentParser(description="Generador Simbólico de Música en Notación ABC")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], required=True, 
                        help="Arquitectura a utilizar para la inferencia")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Ruta al archivo de pesos (.pt), ej: checkpoints/Transformer_best.pt")
    parser.add_argument("--data_path", type=str, default="data/piano-musics-abc-notation.txt", 
                        help="Ruta al dataset original para reconstruir el vocabulario exacto")
    parser.add_argument("--seed", type=str, default='"A"', 
                        help="Texto semilla inicial para impulsar la melodía")
    parser.add_argument("--length", type=int, default=300, 
                        help="Longitud en caracteres de la canción a generar")
    parser.add_argument("--temp", type=float, default=0.8, 
                        help="Temperatura de muestreo (valores bajos = conservador, altos = creativo)")
    parser.add_argument("--out_file", type=str, default=None, 
                        help="Ruta opcional para guardar la salida en un archivo .abc")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Cargando inferencia en dispositivo: {device}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {args.data_path}. Requerido para mapear tokens.")
        
    with open(args.data_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}

    if args.model == "lstm":
        model = LSTMModel(vocab_size).to(device)
    else:
        model = TransformerModel(vocab_size).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    print(f"-> Modelo {args.model.upper()} cargado con éxito. Muestreando melodía...")
    
    raw_song = generate_sample(
        model=model, 
        seed_text=args.seed, 
        char2idx=char2idx, 
        idx2char=idx2char, 
        device=device, 
        length=args.length, 
        temperature=args.temp
    )
    
    final_abc = add_synthetic_header(raw_song)
    
    print("\n" + "="*55)
    print("                 MELODÍA ABC GENERADA")
    print("="*55)
    print(final_abc)
    print("="*55 + "\n")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True) if os.path.dirname(args.out_file) else None
        with open(args.out_file, "w", encoding="utf-8") as out:
            out.write(final_abc)
        print(f"✅ Melodía guardada en disco: {args.out_file}")

if __name__ == "__main__":
    main()