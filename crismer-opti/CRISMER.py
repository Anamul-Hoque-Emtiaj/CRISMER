#!/usr/bin/python

import argparse
import pandas as pd
import numpy as np
from crismer_modules import CRISMER
from utils import *
import os
import pickle
import torch 

__version__ = 'v1.0'
pwd = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(pwd, 'models/change_site_circleseq_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config =config = {
    'num_layers': 2, 
    'num_heads': 4, 
    'number_hidder_layers': 2, 
    'dropout_prob': 0.2, 
    'batch_size': 128, 
    'epochs': 50, 
    'learning_rate': 0.001, 
    'pos_weight': 30, 
    'attn': False,
    "seq_length":20
}
model = CRISPRTransformerModel(config)
model = model.to(device)
model.load_state_dict(torch.load(model_path,weights_only=True))

GENOME = os.path.join(pwd, 'data/hg38.fa')


with open(os.path.join(pwd, 'models/bin_weights.pkl'), 'rb') as f:
    bins, weights = pickle.load(f)
    
scaler_instance = joblib.load("models/minmax_scaler.pkl")


crismer = CRISMER(model = model , bins=bins, weights=weights,ref_genome=GENOME,scaler=scaler_instance)

def cal_score(sgr, tar, ref_genome=GENOME):
    crismer.ref_genome = ref_genome
    return crismer.single_score_(sgr, tar)

def cal_scores(df_in, On='On', Off='Off', ref_genome=GENOME):
    crismer.ref_genome = ref_genome
    df_in['CRISMER_Score'] = crismer.score(df_in)
    return df_in

def cal_spec(df_in, On='On', Off='Off', ref_genome=GENOME):
    crismer.ref_genome = ref_genome
    spec = crismer.spec_per_sgRNA(data_df = df_in, On=On, Off=Off)
    return spec

def cal_casoffinder_spec(sgr, tar, ref_genome, mm=6, dev='G0'):
    crismer.ref_genome = ref_genome
    crismer.casoffinder_mm = mm
    spec = crismer.CasoffinderSpec_(sgr, tar, mm=mm, dev=dev)
    return spec

def opti(tar, ref_genome, threshold=0.77, mm=6, dev='G0'):
    crismer.opti_th = threshold
    crismer.ref_genome = ref_genome
    csv_df = crismer.opti(target=tar, mm=mm, dev=dev)
    return csv_df

# setup the argument parser
def get_parser():
    parser = argparse.ArgumentParser(description="CRISMER Suite "+__version__)
    module_input = parser.add_argument_group("# CRISMER modules")
    module_input.add_argument("method", metavar="<method>", type=str, default='score', 
        help="Select a method to calculate the score(s): \n \
        score: CRISMER-Score of sgRNA-DNA, requires [--sgr, --tar]; \n \
        scores: Batch calculation of CRISMER-Score, requires [--csv], options [--on_item, --off_item, --out]; \n \
        spec: Calculate CRISMER-Spec, on-target sequence must be in the first line, requires [--csv], options [--on_item, --off_item]; \n \
        off_spec: Perform Cas-Offinder search and calculate CRISMER-Spec, requires [--sgr, --tar, --genome], options [--mm, --dev]; \n \
        rescore: Rescoring sgRNAs by CRISMER-Score and CRISMER-Spec, requires [--txt, --genome], options [--mm, --dev, --out]; \n \
        opti: CRISMER-Opti optimization by mutation, requires [--tar, --genome], options [--threshold, --mm, --dev, --out]; \n "
        )
    
    key_settings = parser.add_argument_group('# Key Settings')
    key_settings.add_argument("--sgr", metavar="<seq>", type=str, default=None, 
        help="sgRNA sequence to analyse (23nt, 20nt+PAM)")
    key_settings.add_argument("--tar", metavar="<seq>", type=str, default=None, 
        help="Target DNA sequence to analyse (23nt, 20nt+PAM)")
    key_settings.add_argument("--csv", metavar="<file>", type=str, default=None, 
        help="CSV file containing sgRNA and Target DNA sequences, headers are On and Off, respectively. (spec: On-target sequence must be in the first line)")
    key_settings.add_argument("--txt", metavar="<file>", type=str, default=None, 
        help="TXT file of the designed sgRNAs.")
    key_settings.add_argument("--genome", metavar="<file>", type=str, default=os.path.join(pwd, 'hg38.fa'), 
        help="Path to the file of reference genome. (default: hg38.fa)")

    out_setting = parser.add_argument_group("# Output Settings")
    out_setting.add_argument('--out', metavar="<file>", type=str, default='default', 
        help="Output file name.")

    other_option = parser.add_argument_group("# Other Options")
    other_option.add_argument('--on_item', metavar="<name>", type=str, default='On', 
        help="sgRNA column name of the csv file (default: On)")
    other_option.add_argument('--off_item', metavar="<name>", type=str, default='Off', 
        help="Off-target column name of the csv file (default: Off)")
    other_option.add_argument('--mm', metavar="<number>", type=int, default=6, 
        help="Mismatch tolerance (default: 6)")
    other_option.add_argument('--dev', metavar="<device>", type=str, default='G', 
        help="GPU/CPU device setting, the same as in the CasOffinder (default: G)")
    other_option.add_argument('--threshold', metavar="<float>", type=float, default=0.77, 
        help="The CRISMER-Score threshold for mutated sgRNAs. (default: 0.77)")

    parser.add_argument('--version', action='version', version='CRISMER {}'.format(__version__))

    return parser


# MAIN FUNCTION
def main():
    # Get the necessary arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Method 1: CRISMER-Score
    if args.method == 'score':
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt with NGG PAM'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        score = cal_score(args.sgr, args.tar)
        print('CRISMER-Score: \n' + str(score))
    
    elif args.method == 'scores':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        scores = cal_scores(df_csv, On=args.on_item, Off=args.off_item)
        if args.out == 'default':
            scores.to_csv('CRISMER-Score_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            scores.to_csv(out_name, index=False)
        print('CRISMER-Score calculation done')

    elif args.method == 'spec':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        spec = cal_spec(df_csv, On=args.on_item, Off=args.off_item)
        print('CRISMER-Spec: \n' + str(spec))
    elif args.method == 'off_spec':
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt with NGG PAM'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)

        spec = cal_casoffinder_spec(args.sgr, args.tar, ref_genome=args.genome, mm=args.mm, dev=args.dev)
        print('CRISMER-Spec: \n' + str(spec))
    elif args.method == 'opti':
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)

        opti_result = opti(args.tar, ref_genome=args.genome, threshold=args.threshold, mm=args.mm, dev=args.dev)
        if args.out == 'default':
            opti_result.to_csv('CRISMER-Opti_optimization_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            opti_result.to_csv(out_name, index=False)
        print('CRISMER-Opti optimization done')

    else:
        print('Please choose a correct method')
if __name__== "__main__":
    main()


