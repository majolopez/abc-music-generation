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
    parser.add_argument("--seed", type=str, default="A",
                        help="Texto semilla inicial para impulsar la melodía")
    parser.add_argument("--length", type=int, default=300,
                        help="Longitud en caracteres de la canción a generar")
    parser.add_argument("--temp", type=float, default=0.8,
                        help="Temperatura de muestreo (valores bajos = conservador, altos = creativo)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling. Recomendado: 10 para LSTM, 20 para Transformer "
                             "(evita loops repetitivos observados sin top_k)")
    parser.add_argument("--context_window", type=int, default=100,
                        help="Ventana de contexto usada durante el entrenamiento (debe coincidir)")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Ruta opcional para guardar la salida en un archivo .abc")
    parser.add_argument("--disable_cudnn", action="store_true",
                        help="Usar solo si tu servidor tiene el bug de versiones de cuDNN en conflicto")
    args = parser.parse_args()

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("-> cuDNN deshabilitado (workaround manual solicitado)")

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
        model = TransformerModel(vocab_size, context_window=args.context_window).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print(f"-> Modelo {args.model.upper()} cargado con éxito. "
          f"Muestreando melodía (temp={args.temp}, top_k={args.top_k})...")

    raw_song = generate_sample(
        model=model,
        seed_text=args.seed,
        char2idx=char2idx,
        idx2char=idx2char,
        device=device,
        length=args.length,
        temperature=args.temp,
        top_k=args.top_k,
        context_window=args.context_window,
    )

    final_abc = add_synthetic_header(raw_song)

    print("\n" + "=" * 55)
    print("                 MELODÍA ABC GENERADA")
    print("=" * 55)
    print(final_abc)
    print("=" * 55 + "\n")

    if args.out_file:
        out_dir = os.path.dirname(args.out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as out:
            out.write(final_abc)
        print(f"Melodía guardada en disco: {args.out_file}")


if __name__ == "__main__":
    main()