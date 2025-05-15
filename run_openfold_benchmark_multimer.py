import argparse
import logging
import math
import numpy as np
import os
import shutil
import json
from collections import Counter
import re
import glob  
import sys
from datetime import date
import itertools
import time 

from openfold.utils.script_utils import load_model, parse_fasta, run_model, prep_output, \
    update_timings, relax_protein

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import subprocess 
import pickle

import random
import time
import torch
from torch import nn

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts.utils import add_data_args
import pandas as pd 

from rw_helper_functions import write_timings, remove_files, calc_disordered_percentage
from custom_openfold_utils.pdb_utils import align_and_get_rmsd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./benchmark_multimer.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def eval_model(model, args, config, feature_processor, feature_dict, processed_feature_dict, tag, output_dir):

    print('Tag: %s' % tag)
    os.makedirs(output_dir, exist_ok=True)

    out, inference_time = run_model(model, processed_feature_dict, tag, output_dir, return_inference_time=True)

    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()),
        processed_feature_dict
    )
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
    mean_plddt = np.mean(out["plddt"])
    ptm_score = np.squeeze(out["ptm_score"])
    num_recycles = int(np.squeeze(out["num_recycles"]))

    if "iptm_score" in out:
        iptm_score = np.squeeze(out["iptm_score"])
        weighted_ptm_score = np.squeeze(out["weighted_ptm_score"])
    else:
        iptm_score = None
        weighted_ptm_score = None 

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        args.config_preset,
        args.multimer_ri_gap,
        args.subtract_plddt
    )

    output_name = 'temp'
    model_output_dir_temp = '%s/temp' % output_dir
    os.makedirs(model_output_dir_temp, exist_ok=True)

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if args.cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        model_output_dir_temp, f'{output_name}{unrelaxed_file_suffix}'
    )
 
    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    disordered_percentage = calc_disordered_percentage(unrelaxed_output_path)
    shutil.rmtree(model_output_dir_temp)

    output_name = tag
    model_output_dir = output_dir

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if args.cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        model_output_dir, f'{output_name}{unrelaxed_file_suffix}'
    )
 
    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    logger.info(f"Output written to {unrelaxed_output_path}...")

    return mean_plddt, float(weighted_ptm_score), disordered_percentage, num_recycles, inference_time, unrelaxed_output_path 


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    models_to_run = ['model_1_multimer_v3', 'model_2_multimer_v3', 'model_3_multimer_v3', 'model_4_multimer_v3', 'model_5_multimer_v3']
    jax_param_path_dict = {} 
    for m in models_to_run:
        jax_param_path_dict[m] = ''.join((args.jax_param_parent_path, '/params_', m, '.npz'))
    
    print('MODEL LIST:')
    print(jax_param_path_dict)

    alignment_dir = args.alignment_dir
    file_id = [x[0] for x in os.walk('.')]
    file_id = os.listdir('./')
    if len(file_id) > 1:
        raise ValueError("should only be a single directory under %s" % alignment_dir)
    else:
        file_id = file_id[0] #e.g 1xyz-1xyz
    alignment_dir_w_file_id = '%s/%s' % (alignment_dir, file_id)
    print("alignment directory with file_id: %s" % alignment_dir_w_file_id)
    
    if args.fasta_file is None:
        pattern = "%s/*.fasta" % alignment_dir_w_file_id
        files = glob.glob(pattern, recursive=True)
        if len(files) == 1:
            fasta_file = files[0]
        else: 
            raise FileNotFoundError("Multiple .fasta files found in alignment_dir -- should only be one")
    else:
        fasta_file = args.fasta_file

    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seqs = parse_fasta(fasta_data)
    print(seqs)

    config_dict = {}
    for m in models_to_run:
        config_name = m 
        config = model_config(config_name, long_sequence_inference=args.long_sequence_inference)

        if args.recycle_wo_early_stopping:
            config.model.recycle_early_stop_tolerance = -1 #do full recycling 
            config.data.common.max_recycling_iters = args.max_recycling_iters
            num_recycles_str = str(args.max_recycling_iters+1)
        else:
            num_recycles_str = 'early_stopping'

        if m not in config_dict:
            config_dict[m] = config
            if(args.trace_model):
                if(not config.data.predict.fixed_size):
                    raise ValueError(
                        "Tracing requires that fixed_size mode be enabled in the config"
                    )

    output_dir = '%s/benchmark/num_recycles=%s' % (args.output_dir_base, num_recycles_str)    
    output_dir = os.path.abspath(output_dir)
    print('Output Directory: %s' % output_dir)
    os.makedirs(output_dir, exist_ok=True)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config_dict[list(config_dict.keys())[0]].data)
     
    pattern = "%s/features.pkl" % alignment_dir_w_file_id
    files = glob.glob(pattern, recursive=True)
    if len(files) == 1:
        features_output_path = files[0]
    print('features.pkl path: %s' % features_output_path)
    if os.path.isfile(features_output_path):
        feature_dict = np.load(features_output_path, allow_pickle=True) #this is used for all predictions, so this assumes you are predicting a single sequence 
    else:
        raise FileNotFoundError('%s/features.pkl not found' % alignment_dir)

    print(feature_dict)
    num_chains = int(feature_dict['assembly_num_chains']) 
    print('NUM CHAINS: %d' % num_chains)
 
    #this uses the standard inference configuration with no dropout 
    model_dict = {} 
    for m in models_to_run: 
        model = load_model(config_dict[m], args.model_device, None, jax_param_path_dict[m], enable_dropout=False)
        if m not in model_dict:
            model_dict[m] = model

   
    if args.skip_initial_pred_phase:

        initial_pred_dir = '%s/initial_pred' %  output_dir
        initial_pred_info_fname = '%s/initial_pred_info.pkl' % initial_pred_dir
        seed_fname = '%s/seed.txt' % initial_pred_dir
        if os.path.exists(initial_pred_info_fname) and os.path.exists(seed_fname):
            with open(initial_pred_info_fname, 'rb') as f:
                initial_pred_path_dict = pickle.load(f)
            random_seed = int(np.loadtxt(seed_fname))
            np.random.seed(random_seed)
            torch.manual_seed(random_seed + 1)
            print('SKIPPING INITIAL PRED PHASE')
        else:
            raise FileNotFoundError('%s not found' % initial_pred_info_fname)

        #process features after updating seed 
        print('PROCESSING FEATURES')
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=True
        )
        processed_feature_dict = {
            k:torch.as_tensor(v, device=args.model_device)
            for k,v in processed_feature_dict.items()
        }

    else:
 
        print('PROCESSING FEATURES')
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=True
        )
        processed_feature_dict = {
            k:torch.as_tensor(v, device=args.model_device)
            for k,v in processed_feature_dict.items()
        }

        t0 = time.perf_counter()
        initial_pred_dir = '%s/initial_pred' %  output_dir
        initial_pred_path_dict = {}         
        conformation_info_dict = {}

        for i in range(0,len(models_to_run)):
            model_name = models_to_run[i]
            tag = model_name
            print('RUNNING model %s' % model_name)
            mean_plddt_initial, weighted_ptm_score_initial,  disordered_percentage_initial, _, _, pdb_path_initial = eval_model(model_dict[model_name], args, config_dict[model_name], feature_processor, feature_dict, processed_feature_dict, tag, initial_pred_dir)    
            print('pLDDT: %.3f, IPTM_PTM SCORE: %.3f, disordered percentage: %.3f, INITIAL' % (mean_plddt_initial, weighted_ptm_score_initial, disordered_percentage_initial)) 
            conformation_info_dict[model_name] = (pdb_path_initial, mean_plddt_initial, weighted_ptm_score_initial, disordered_percentage_initial) 
            initial_pred_path_dict[model_name] = pdb_path_initial 
            print(initial_pred_path_dict)   

        run_time = time.perf_counter() - t0
        timing_dict = {'initial_pred': run_time} 
        write_timings(timing_dict, output_dir, 'inital_pred')

        initial_pred_info_fname = '%s/initial_pred_info.pkl' % initial_pred_dir
        with open(initial_pred_info_fname, 'wb') as f:
            pickle.dump(initial_pred_path_dict, f)

        conformation_info_fname = '%s/conformation_info.pkl' % initial_pred_dir
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info_dict, f)

        seed_fname = '%s/seed.txt' % initial_pred_dir
        np.savetxt(seed_fname, [random_seed], fmt='%d')


    #this uses the standard inference configuration with dropout 
    model_dict = {} 
    for m in models_to_run: 
        model = load_model(config_dict[m], args.model_device, None, jax_param_path_dict[m], enable_dropout=True)
        if m not in model_dict:
            model_dict[m] = model


    for i in range(0,len(models_to_run)):

        t0 = time.perf_counter()

        model_name = models_to_run[i]
        pred_output_dir = '%s/source=%s' %  (output_dir, model_name)   
        pdb_path_initial = initial_pred_path_dict[model_name] 

        pdb_files = glob.glob('%s/*.pdb' % pred_output_dir)
        if len(pdb_files) >= args.num_predictions_per_model:
            if args.overwrite_pred:
                print('removing pdb files in %s' % pred_output_dir)
                remove_files(pdb_files)
            else:
                print('SKIPPING PREDICTION FOR: %s --%d files already exist--' % (pred_output_dir, len(pdb_files)))
                continue 
        elif len(pdb_files) > 0: #incomplete job
            print('removing pdb files in %s' % pred_output_dir)
            remove_files(pdb_files)

        conformation_info_dict = {}
        conformation_info = [] 

        for j in range(0,args.num_predictions_per_model):

            tag = 'pred_%d' % (j+1)
            print('RUNNING model %s, pred %d' % (model_name,j))
            mean_plddt, weighted_ptm_score,  disordered_percentage, num_recycles, inference_time, pdb_path = eval_model(model_dict[model_name], args, config_dict[model_name], feature_processor, feature_dict, processed_feature_dict, tag, pred_output_dir)   
            print('pLDDT: %.3f, IPTM_PTM SCORE: %.3f, disordered percentage: %.3f, num recycles: %d' % (mean_plddt, weighted_ptm_score, disordered_percentage, num_recycles)) 

            rmsd = align_and_get_rmsd(pdb_path_initial, pdb_path)            
            conformation_info.append((pdb_path, rmsd, mean_plddt, weighted_ptm_score, disordered_percentage, num_recycles, inference_time))

        conformation_info_dict[model_name] = conformation_info
        conformation_info_output_dir = pred_output_dir
        conformation_info_fname = '%s/conformation_info.pkl' % conformation_info_output_dir
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info_dict, f)

        inference_key = 'inference_%d' % i
        run_time = time.perf_counter() - t0
        timing_dict = {inference_key: run_time} 
        write_timings(timing_dict, output_dir, inference_key)


           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_file", type=str, default=None,
        help="Path to FASTA file, one sequence per file. By default assumes that .fasta file is located in alignment_dir "
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--alignment_dir", type=str, required=True,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir_base", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default=None,
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--jax_param_parent_path", type=str, default=None,
        help="""Parent ath to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument(
        "--num_predictions_per_model", type=int, default=50
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--cif_output", action="store_true", default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    parser.add_argument(
        "--skip_initial_pred_phase", action="store_true", default=False
    )    
    parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )
    parser.add_argument(
        "--recycle_wo_early_stopping", action="store_true", default=False
    )
    parser.add_argument(
        "--max_recycling_iters", type=int, default=19
    )



    add_data_args(parser)
    args = parser.parse_args()

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)

