import argparse
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, HubertModel
from tqdm import tqdm

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False


# ----------------------------------
# cosine distance helper
# ----------------------------------
def fast_cosine_dist(src: torch.Tensor, pool: torch.Tensor):
    src_norm = F.normalize(src, p=2, dim=-1)
    pool_norm = F.normalize(pool, p=2, dim=-1)
    return 1 - torch.matmul(src_norm, pool_norm.transpose(0, 1))


# ----------------------------------
# mHuBERT feature extractor
# ----------------------------------
def load_mhubert(device):
    feat_extractor = AutoFeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
    model = HubertModel.from_pretrained("utter-project/mHuBERT-147", output_hidden_states=True).to(device)
    model.eval()
    return model, feat_extractor


@torch.no_grad()
def extract_mhubert_feats(wav_path: Path, model, extractor, device, layer: int):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.squeeze(0).to(device)

    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    feats = out.hidden_states[layer]  # (1, T, dim)

    return feats.squeeze(0).cpu()  # (T, dim)


# ----------------------------------
# main prematch logic
# ----------------------------------
def main(args):
    device = torch.device(args.device)

    print("Loading metadata TSV...")
    df = pd.read_csv(args.tsv, sep="\t")
    df["path"] = df["path"].apply(lambda x: Path(args.dataset_root) / x)

    print("Loading mHuBERT model...")
    model, extractor = load_mhubert(device)

    print("Running prematcher speaker by speaker...")
    out_root = Path(args.out_dir)
    out_root.mkdir(exist_ok=True, parents=True)

    for speaker in df["client_id"].unique():

        print(f"Processing speaker: {speaker}")

        spk_df = df[df.client_id == speaker]

        # ---- extract speaker features once ----
        speaker_feats = {}

        for _, row in spk_df.iterrows():
            path = row.path
            if not path.exists():
                continue
            try:
                speaker_feats[path] = extract_mhubert_feats(
                    path, model, extractor, device, args.layer
                )
            except Exception as e:
                print(f"[ERROR] {path}: {e}")
                continue

        paths = list(speaker_feats.keys())

        # ---- prematch inside speaker ----
        for src_path in paths:

            source_feats = speaker_feats[src_path]

            pool_feats = [
                speaker_feats[p] for p in paths if p != src_path
            ]

            if len(pool_feats) == 0:
                continue

            pool_feats = torch.cat(pool_feats, dim=0)

            dists = fast_cosine_dist(source_feats, pool_feats)
            best = dists.topk(k=args.k, dim=-1, largest=False)
            matched = pool_feats[best.indices].mean(dim=1)

            rel = src_path.relative_to(args.dataset_root)
            out_path = out_root / rel.with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(matched.half(), out_path)

        # ---- clear speaker memory ----
        del speaker_feats
        torch.cuda.empty_cache()

    print("Prematching finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="root where wav paths are relative")
    parser.add_argument("--tsv", required=True, help="metadata TSV with path and speaker columns")
    parser.add_argument("--out_dir", required=True, help="where to store prematched features (.pt)")
    parser.add_argument("--layer", type=int, default=12, help="mHuBERT layer to use")
    parser.add_argument("--k", type=int, default=4, help="kNN topk")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
