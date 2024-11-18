"""A script for performing inference using a model
fine-tuned on one of the Absolut! datasets for Parkinson
et al. 2024. It takes as input a filepath to
the fine-tuned inference model state dict and
an output path."""
import argparse
import numpy as np
import torch
from evodiff.generate import generate_oaardm
from evodiff.pretrained import OA_DM_38M, OA_DM_640M


def parse_args():
    """Constructs a simple argparser and returns the parsed
    args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('state_dict_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--large_model', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.large_model:
        model, collater, tokenizer, _ = OA_DM_640M()
    else:
        model, collater, tokenizer, _ = OA_DM_38M()

    model.load_state_dict(torch.load(args.state_dict_path))
    model = model.eval().to("cuda")
    _ = torch.manual_seed(0)
    np.random.seed(0)

    generated_sequences, batch_size = [], 10

    # Use the same desired binder lengths as the RESP study.
    for seqlen in [11,13,15,17,19]:
        for _ in range(0,200,batch_size):
            tokenized_sample, seqbatch = \
                    generate_oaardm(model, tokenizer, seqlen,
                        batch_size=batch_size, device='cuda')
            generated_sequences += seqbatch

    with open(args.output_path, "w+",
              encoding="utf-8") as fhandle:
        for i, seq in enumerate(generated_sequences):
            fhandle.write(f"{i}\t{seq}\n")


if __name__ == "__main__":
    main()
