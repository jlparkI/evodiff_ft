import numpy as np
import dms.utils
from dms.pretrained import MSA_D3PM_UNIFORM_RANDSUB, MSA_D3PM_UNIFORM_MAXSUB, MSA_D3PM_BLOSUM_RANDSUB, \
    MSA_D3PM_BLOSUM_MAXSUB, MSA_OA_AR_RANDSUB, MSA_OA_AR_MAXSUB, ESM_MSA_1b
from dms.losses import D3PMCELoss
from sequence_models.losses import MaskedCrossEntropyLossMSA
import torch
from tqdm import tqdm
import pandas as pd
from analysis.plot import plot_perp_group_masked, plot_perp_group_d3pm
import argparse
import os

def main():
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='D3PM_BLOSUM_38M',
                        help='Choice of: msa_d3pm_uniform_randsub, msa_d3pm_uniform_maxsub,\
                         msa_d3pm_blosum_randsub, msa_d3pm_blosum_maxsub,\
                         msa_oa_ar_randsub, msa_oa_ar_maxsub, esm_msa_1b')
    args = parser.parse_args()

    save_name = args.model_type

    if args.model_type=='msa_d3pm_uniform_randsub':
        checkpoint = MSA_D3PM_UNIFORM_RANDSUB()
        selection_type='random'
    elif args.model_type=='msa_d3pm_uniform_maxsub':
        checkpoint = MSA_D3PM_UNIFORM_MAXSUB()
        selection_type='MaxHamming'
    elif args.model_type=='msa_d3pm_blosum_randsub':
        checkpoint = MSA_D3PM_BLOSUM_RANDSUB()
        selection_type = 'random'
    elif args.model_type=='msa_d3pm_blosum_maxsub':
        checkpoint = MSA_D3PM_BLOSUM_MAXSUB()
        selection_type='MaxHamming'
    elif args.model_type=='msa_oa_ar_randsub':
        checkpoint = MSA_OA_AR_RANDSUB()
        selection_type='random'
    elif args.model_type=='msa_oa_ar_maxsub':
        checkpoint = MSA_OA_AR_MAXSUB()
        selection_type = 'MaxHamming'
    elif args.model_type=='esm_msa_1b':
        checkpoint = ESM_MSA_1b()
        selection_type = 'MaxHamming'
    else:
        print("Please select valid model, i don't understand:", args.model_type)
    #print(checkpoint)
    # Def read seqs from fasta
    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/data/data/data/'
    except:
        data_top_dir = 'data/'

    num_seqs=20000

    data = dms.utils.get_valid_msas(data_top_dir, data_dir='openfold/', selection_type=selection_type, n_sequences=64, max_seq_len=512,
                   out_path='../DMs/ref/')

    losses = []
    n_tokens = []
    time_loss_data = []
    for i in tqdm(range(num_seqs)): #len(data))):
        r_idx = np.random.choice(len(data))
        sequence = [data[r_idx]]
        t, loss, tokens = sum_nll_mask(sequence, checkpoint)
        if not np.isnan(loss): #esm-1b predicts nans at large % mask
            #print(loss, tokens)
            losses.append(loss)
            n_tokens.append(tokens)
            time_loss_data.append([t, loss, tokens])
        if i % 1000 == 0:
            ll = -sum(losses) / sum(n_tokens)
            perp = np.exp(-ll)
            print(i, "samples, perp:", np.mean(perp))
    print("Final test perp:", np.exp(sum(losses)/sum(n_tokens)))
    df = pd.DataFrame(time_loss_data, columns=['time', 'loss', 'tokens'])
    if checkpoint[-1] == 'd3pm':
        plot_perp_group_d3pm(df, save_name)
    else:
        plot_perp_group_masked(df, save_name)

def sum_nll_mask(sequence, checkpoint):
    model, collater, tokenizer, scheme = checkpoint
    model.eval().cuda() # Use model.eval() if using CPU

    # D3PM Collater returns; src, src_one_hot, timesteps, tokenized, tokenized_one_hot, Q, Q_bar, q_x
    if scheme == 'd3pm':
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = collater(sequence)
        input_mask = (src != tokenizer.pad_id).float() # placeholder
        input_mask = input_mask.cuda()
        timestep = timestep.cuda()
    elif scheme == 'mask':
        src, tgt, mask = collater(sequence)
        input_mask = (src != tokenizer.pad_id).float() # placeholder, should be no pads since not batching
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    elif scheme == 'esm-mask':
        src, tgt, mask = collater(sequence)
        input_mask = (src != tokenizer.padding_idx).float()  # placeholder, should be no pads since not batching
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    src = src.cuda()     # Comment all variable.cuda() lines if using CPU
    tgt = tgt.cuda()
    with torch.no_grad():
        #print(timestep)
        if scheme == 'd3pm':
            outputs = model(src, timestep) # outputs are x_tilde_0 (predicted tgt)
        elif scheme == 'esm-mask':
            outputs = model(src, repr_layers=[33], return_contacts=True)
            outputs = outputs["logits"]
        else:
            outputs = model(src)

    # Get loss (NLL ~= CE)
    if scheme == 'd3pm':
        loss_func = D3PMCELoss(reduction='sum',tokenizer=tokenizer, sequences=False)
        #print(outputs.shape, tgt.shape)
        nll_loss = loss_func(outputs, tgt, input_mask)
        #print(nll_loss)
        t_out=timestep
        tokens_msa = tgt.squeeze().shape
        tokens = tokens_msa[0]*tokens_msa[1]
    elif scheme == 'mask' or scheme == 'esm-mask':
        if scheme == 'esm-mask':
            loss_func = MaskedCrossEntropyLossMSA(ignore_index=tokenizer.padding_idx, reweight=False)
        else:
            loss_func = MaskedCrossEntropyLossMSA(ignore_index=tokenizer.pad_id, reweight=False)
        #print(outputs.shape, tgt.shape, mask.shape)
        ce_loss, nll_loss = loss_func(outputs, tgt, mask, input_mask) # returns a sum
        #print(nll_loss, mask.sum())
        tokens = mask.sum().item()
        t_out = tokens / int(tgt.squeeze().shape[0]*tgt.squeeze().shape[1])
        #print(t_out)
    return t_out, nll_loss.item(), tokens # return timestep sampled (or % masked), sum of losses, and sum of tokens

if __name__ == '__main__':
    main()