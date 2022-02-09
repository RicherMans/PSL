import pandas as pd
import argparse
import torch
import models
import librosa
from einops import rearrange

LABELS = pd.read_csv('features/storage/class_indicies/class_labels_indices.csv',sep=',')


def overlapping_windows(x:torch.Tensor, win_size:int, hop_size:int):
    dx = rearrange(x, 'b t -> b 1 1 t')
    dx = torch.nn.functional.unfold(dx, kernel_size=(1,win_size), stride=(1,hop_size))
    n_chunks = dx.shape[-1]
    dx = rearrange(dx, 'b t chunks -> (chunks b) t')
    return dx, n_chunks


LABELTYPES = {
        'standard': lambda idx: LABELS.iloc[idx]['display_name'],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath')
    parser.add_argument('inputaudio', type=str, nargs='+')
    parser.add_argument('--label',default='standard', choices=list(LABELTYPES.keys()))
    parser.add_argument('-m','--model', default='MobileNetV2_Decision_MeanPool')
    parser.add_argument('--win', default=None, type=int)
    parser.add_argument('--hop', default=None, type=int)
    parser.add_argument('-k', default=5, type=int)
    args = parser.parse_args()

    LABELFUN = LABELTYPES[args.label]

    model = getattr(models, args.model)(outputdim=527)
    model_dump = torch.load(args.modelpath,map_location='cpu')
    if 'model' in model_dump:
        model_dump= model_dump['model']
    model = models.load_pretrained(model, model_dump).eval()

    for audio in args.inputaudio:
        wav, _ = librosa.load(audio, sr=16000)
        with torch.no_grad():
            x = torch.as_tensor(wav).unsqueeze(0)
            if args.win is not None and x.shape[-1] > args.win:
                x, _ = overlapping_windows(x, args.win, args.hop)
                print(_)
                y, _ = model(x)
                y = y.max(0)[0]
            else:
                y, _ = model(x)
            prob, lab = y.squeeze(0).topk(args.k)
            prob = prob.numpy()
            lab = lab.numpy()
            lab = [LABELFUN(l) for l in lab]
            print(f"{audio}")
            for a, b in zip(prob, lab):
                print(f"{b:<15} {a*100:.2f}")

if __name__ == "__main__":
    main()

