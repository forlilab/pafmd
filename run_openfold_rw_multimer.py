import argparse
import logging
import math
import numpy as np
import pandas as pd 
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
import ml_collections as mlc
from typing import Tuple, List, Mapping, Optional, Sequence, Any, MutableMapping, Union

from openfold.utils.script_utils import load_model_w_intrinsic_param, parse_fasta, run_model_w_intrinsic_dim, prep_output, \
    update_timings, relax_protein

import subprocess 
import pickle

import random
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

from random_corr_sap import gen_randcorr_sap
from custom_openfold_utils.pdb_utils import align_and_get_rmsd
import rw_helper_functions

FeatureDict = MutableMapping[str, np.ndarray]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./rw_multimer.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


rw_finetune_path = './rw_finetune.py'

TRACING_INTERVAL = 50
SEQ_LEN_EXTRAMSA_THRESHOLD = 600 #if length of seq > 600, extra_msa is reduced so backprop doesn't crash
asterisk_line = '******************************************************************************'


def eval_model(model, config, intrinsic_parameter, epsilon, epsilon_scaling_factor, feature_processor, feature_dict, processed_feature_dict, tag, output_dir, phase, args):

    os.makedirs(output_dir, exist_ok=True)

    model.intrinsic_parameter = nn.Parameter(torch.tensor(intrinsic_parameter, dtype=torch.float).to(args.model_device))
    model.epsilon = nn.Parameter(torch.tensor(epsilon, dtype=torch.float).to(args.model_device)) 
    model.epsilon_scaling_factor = nn.Parameter(torch.tensor(epsilon_scaling_factor, dtype=torch.float).to(args.model_device))
    out, inference_time = run_model_w_intrinsic_dim(model, processed_feature_dict, tag, output_dir, return_inference_time=True)

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

    disordered_percentage = rw_helper_functions.calc_disordered_percentage(unrelaxed_output_path)
    accept_conformation = rw_helper_functions.accept_criteria(mean_plddt, disordered_percentage, args.mean_plddt_threshold, args.disordered_percentage_threshold)
    shutil.rmtree(model_output_dir_temp)

    if phase == 'initial': #i.e the initial AF prediction from the original MSA
        output_name = tag
        model_output_dir = output_dir
    else:
        if accept_conformation:
            output_name = '%s-A' % tag  
            model_output_dir = '%s/ACCEPTED' % output_dir
            os.makedirs(model_output_dir, exist_ok=True)
        else:
            output_name = '%s-R' % tag 
            model_output_dir = '%s/REJECTED' % output_dir 
            os.makedirs(model_output_dir, exist_ok=True) 

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

    if args.relax_conformation:
        # Relax the prediction.
        logger.info(f"Running relaxation on {unrelaxed_output_path}...")
        relax_protein(config, args.model_device, unrelaxed_protein, model_output_dir, output_name,
                      args.cif_output)

    if args.save_outputs and accept_conformation:
        embeddings_output_dir = '%s/embeddings' % model_output_dir
        os.makedirs(embeddings_output_dir, exist_ok=True)

        output_dict_path = os.path.join(
            embeddings_output_dir, f'{output_name}_output_dict.pkl'
        )
        with open(output_dict_path, "wb") as fp:
            pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Model embeddings written to {output_dict_path}...")


    return mean_plddt, float(weighted_ptm_score), disordered_percentage, num_recycles, inference_time, accept_conformation, unrelaxed_output_path 


def update_config(
    config: mlc.ConfigDict, 
    seqs: List[str], 
    num_chains: int, 
    args: argparse.Namespace,
):
    """Updates the config dictionary by adding the appropriate chain masks 
       and updating the appropriate recycling parameters. 
    """

    if args.recycle_wo_early_stopping:
        config.model.recycle_early_stop_tolerance = -1 #do full recycling 
        config.data.common.max_recycling_iters = args.max_recycling_iters
        num_recycles_str = str(args.max_recycling_iters+1)
    else:
        num_recycles_str = 'early_stopping'

    if config.model.use_chainmask:
        total_seq_len = sum(len(s) for s in seqs)
        mask_all = [] 
        chain_mask_row_all = []
        chain_mask_col_all = [] 
        for i,curr_seq in enumerate(seqs):
            mask = torch.zeros(total_seq_len)
            mask_start_pos = sum([len(s) for s in seqs[0:i]])
            mask_end_pos = mask_start_pos+len(curr_seq)
            mask[mask_start_pos:mask_end_pos] = 1. 
            mask_all.append(mask)
            chain_mask_row = mask.unsqueeze(1).to('cuda').to(dtype=torch.bfloat16)
            chain_mask_col = mask.reshape(mask.shape[0],1,1).to('cuda').to(dtype=torch.bfloat16) #this is for col attention (where s,r,c -> r,s,c)
            chain_mask_row_all.append(chain_mask_row)
            chain_mask_col_all.append(chain_mask_col)

        chain_mask_row_all = torch.stack(chain_mask_row_all)
        chain_mask_col_all = torch.stack(chain_mask_col_all)

        chain_mask_row_all.requires_grad_(False)
        chain_mask_col_all.requires_grad_(False)
        config.model.extra_msa.extra_msa_stack.chain_mask_row = chain_mask_row_all
        config.model.extra_msa.extra_msa_stack.chain_mask_col = chain_mask_col_all
        config.model.evoformer_stack.chain_mask_row = chain_mask_row_all 
        config.model.evoformer_stack.chain_mask_col = chain_mask_col_all
        config.rw_finetuning.num_chains = num_chains
    else:
        config.model.extra_msa.extra_msa_stack.chain_mask_row = None
        config.model.extra_msa.extra_msa_stack.chain_mask_col = None
        config.model.evoformer_stack.chain_mask_row = None 
        config.model.evoformer_stack.chain_mask_col = None 
 
    return config


def get_aligned_models_info(
    models_to_run: List[Any], 
    initial_pred_path_dict: Mapping[str, str],
    file_id: str, 
    args: argparse.Namespace
):
    """Gets a list of <source,target> models to use for the gradient descent phase of
       the pipeline.  

       For each pair of models, we calculate the RMSD between their predicted structures. 
       For each model, we then find the corresponding model with the maximum RMSD between
       their respective predictions. This pair of models (i.e <source,target>) is then used 
       during the gradient descent phase to learn how to optimize epsilon to align source 
       with target. 
    """
    
    aligned_models_max_rmsd_dict = {} #maps each model to the  model with the most divergent prediction (in terms of RMSD)  

    for i,model_name_i in enumerate(models_to_run):
        pred_i_path = initial_pred_path_dict[model_name_i]
        for j,model_name_j in enumerate(models_to_run):
            if i != j:        
                pred_j_path = initial_pred_path_dict[model_name_j]
                pred_j_fname = '%s.pdb' % file_id #during training, expected filename is without chain_id (i.e 1xyz-1xyz) 
                aligned_output_dir = '%s/%s/%s/initial_pred_aligned-%s/%s' % (args.output_dir_base, 'rw', args.module_config, model_name_i, model_name_j)
                pred_j_aligned_path = os.path.join(aligned_output_dir,pred_j_fname)
                os.makedirs(aligned_output_dir, exist_ok=True)
                shutil.copy(pred_j_path, pred_j_aligned_path)
                rmsd = align_and_get_rmsd(pred_i_path, pred_j_aligned_path)
                logger.info('RMSD between model %d and model %d: %.3f' % (i+1, j+1, rmsd))
                model_ij_info = (model_name_i,model_name_j,pred_i_path,pred_j_aligned_path,rmsd)

                if i in aligned_models_max_rmsd_dict:
                    max_rmsd = aligned_models_max_rmsd_dict[i][-1]
                    if rmsd > max_rmsd:
                        aligned_models_max_rmsd_dict[i] = model_ij_info 
                else:
                    aligned_models_max_rmsd_dict[i] = model_ij_info

    aligned_models_info = [aligned_models_max_rmsd_dict[key] for key in sorted(aligned_models_max_rmsd_dict.keys())]
    aligned_models_info = aligned_models_info[0:args.num_training_conformations]

    logger.info('ALIGNED MODELS INFO:')    
    logger.info(aligned_models_info)

    return aligned_models_info


def finetune_wrapper(
    aligned_models_info: Tuple[Any,...],
    jax_param_path_dict: Mapping[str, str],
    alignment_dir_wo_file_id: str,
    output_dir: str,
    args: argparse.Namespace,
):

    model_name_source = aligned_models_info[0] #this is model being used for training 
    model_name_target = aligned_models_info[1] 
    source_pdb_path = aligned_models_info[2]
    target_pdb_path = aligned_models_info[3] 
    
    logger.info('ON MODEL: %s' % model_name_source)
    logger.info(aligned_models_info)
    source_str = 'source=%s' % model_name_source #corresponds to model_x_multimer_v3 
    target_str = 'target=%s' % model_name_target
    fine_tuning_save_dir = '%s/training/%s/%s' % (output_dir, source_str, target_str)

    arg1 = '--train_data_dir=%s' % target_pdb_path[0:target_pdb_path.rindex('/')]
    arg2 = '--train_alignment_dir=%s' % alignment_dir_wo_file_id
    arg3 = '--fine_tuning_save_dir=%s' % fine_tuning_save_dir
    arg4 = '--template_mmcif_dir=%s' % args.template_mmcif_dir
    arg5 = '--max_template_date=%s' % date.today().strftime("%Y-%m-%d")
    arg6 = '--gpus=1'
    arg7 = '--resume_from_jax_params=%s' % jax_param_path_dict[model_name_source] 
    arg8 = '--resume_model_weights_only=True'
    arg9 = '--config_preset=multimer-rw_finetuning-SAID-all-%s' % model_name_source #fine tune w.r.t vanila model (i.e no chainmask) 
    arg10 = '--module_config=%s' % args.module_config
    arg11 = '--hp_config=%s' % args.train_hp_config
    arg12 = '--precision=bf16'
    arg13 = '--save_structure_output'

    script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12] 
    if args.save_training_conformations:       
        script_arguments.append(arg13)

    cmd_to_run = ["python", rw_finetune_path] + script_arguments
    cmd_to_run_str = s = ' '.join(cmd_to_run)
    logger.info("RUNNING GRADIENT DESCENT WITH SOURCE %s AND TARGET %s" % (source_pdb_path,target_pdb_path))
    logger.info(asterisk_line)
    logger.info("RUNNING THE FOLLOWING COMMAND:")
    logger.info(cmd_to_run_str)
    subprocess.run(cmd_to_run)



def propose_new_epsilon_vanila_rw(
    intrinsic_dim: int, 
    cov_type: str, 
    L: np.ndarray = None, 
    sigma: np.ndarray = None
):
    """Generates an epsilon vector drawn from a normal distribution with 0 mean 
       and covariance specified according by the user. 
    """

    if cov_type == 'spherical':
        epsilon = np.random.standard_normal(intrinsic_dim)
        epsilon = epsilon/np.linalg.norm(epsilon)
        #intrinsic_param_proposed = intrinsic_param_curr + epsilon*epsilon_scaling_factor
    elif cov_type == 'diagonal':
        if sigma is None:
            raise ValueError("Sigma is missing. It must be provided if cov_type=diagonal")
        epsilon = np.squeeze(np.random.normal(np.zeros(intrinsic_dim), sigma, (1,intrinsic_dim)))
        #intrinsic_param_proposed = intrinsic_param_curr + epsilon*epsilon_scaling_factor
    elif cov_type == 'full':
        if L is None:
            raise ValueError("Cholesky decomposition is missing. It must be provided if cov_type=full")
        x = np.random.standard_normal((intrinsic_dim, 1))
        epsilon = np.dot(L, x).squeeze()
        #intrinsic_param_proposed = intrinsic_param_curr + epsilon*epsilon_scaling_factor 
    
    return epsilon



def run_rw_multimer(
    initial_pred_path: str, 
    intrinsic_dim: int, 
    rw_type: str, 
    rw_hp_dict: Mapping[str, List[float]], 
    num_total_steps: int, 
    L: np.ndarray, 
    cov_type: str, 
    model: nn.Module, 
    config: mlc.ConfigDict, 
    feature_processor: feature_pipeline.FeaturePipeline, 
    feature_dict: FeatureDict, 
    processed_feature_dict: FeatureDict, 
    output_dir: str, 
    phase: str, 
    args: argparse.Namespace,
    save_intrinsic_param: bool = False, 
    terminate_after_first_rejection: bool = False
):
    """Generates new conformations by modifying parameter weights via a random walk on episilon. 
    
       For a given perturbation to epsilon, model weights are updated via fastfood transform. 
       The model is then evaluated and the output conformation is either accepted or rejected
       according to the acceptance criteria. If the conformation is rejected, epsilon is
       reinitialized to the zero vector and the process repeats until num_total_steps have
       been run.  
    """

    base_tag = '%s_train-%s_rw-%s' % (args.module_config, args.train_hp_config, args.rw_hp_config)
    base_tag = base_tag.replace('=','-')

    intrinsic_param_initial = np.zeros(intrinsic_dim)
    epsilon_initial = np.zeros(intrinsic_dim)
    epsilon_prev = epsilon_initial
 
    num_rejected_steps = 0
    num_accepted_steps = 0  
    iter_n = 0 
    curr_step_iter_n = 0
    curr_step_aggregate = 0 
    state_history = [] 
    conformation_info = [] 

    while (num_rejected_steps+num_accepted_steps) < num_total_steps:

        if rw_type == 'vanila':
            epsilon_proposed = propose_new_epsilon_vanila_rw(intrinsic_dim, cov_type, L)

        epsilon_cummulative = epsilon_prev+epsilon_proposed

        curr_tag = '%s_iter%d_step-iter%d_step-agg%d' % (base_tag, iter_n, curr_step_iter_n, curr_step_aggregate) 
        mean_plddt, weighted_ptm_score, disordered_percentage, num_recycles, inference_time, accept_conformation, pdb_path_rw = eval_model(model, config, intrinsic_param_initial, epsilon_cummulative, rw_hp_dict['epsilon_scaling_factor'], feature_processor, feature_dict, processed_feature_dict, curr_tag, output_dir, phase, args)    
        state_history.append(accept_conformation)
        logger.info('pLDDT: %.3f, IPTM_PTM SCORE: %.3f, disordered percentage: %.3f, num recycles: %d, step: %d' % (mean_plddt, weighted_ptm_score, disordered_percentage, num_recycles, curr_step_aggregate)) 

        if accept_conformation:
            logger.info('STEP %d: ACCEPTED' % curr_step_aggregate)
            epsilon_prev = epsilon_cummulative #x_i = x_i-1 + eps_i*s --> = s*sum(eps_i)
            curr_step_iter_n += 1
            num_accepted_steps += 1
            rmsd = align_and_get_rmsd(initial_pred_path, pdb_path_rw)
            if save_intrinsic_param:
                conformation_info.append((pdb_path_rw, rmsd, mean_plddt, weighted_ptm_score, disordered_percentage, num_recycles, inference_time, epsilon_cummulative, rw_hp_dict['epsilon_scaling_factor']))
            else:
                conformation_info.append((pdb_path_rw, rmsd, mean_plddt, weighted_ptm_score, disordered_percentage, num_recycles, inference_time, None, None))
        else:
            logger.info('STEP %d: REJECTED' % curr_step_aggregate)
            if terminate_after_first_rejection:
                return state_history, conformation_info
            else:
                epsilon_prev = epsilon_initial
                iter_n += 1
                curr_step_iter_n = 0 
                num_rejected_steps += 1 

        curr_step_aggregate += 1

    return state_history, conformation_info


def get_new_scaling_factor_candidates(
    rw_hp_acceptance_rate_dict: Mapping[Tuple[float, ...], float],
    rw_hp_config_data: Mapping[str, Any]
):
    """Based on the acceptance rate of other scaling factor combinations, 
       generates new scaling factor candidates to test.

       We begin by finding an upper and lower bound for the scaling factor. 
       To find the lower bound, we find the largest epsilon scaling factor
       greater than the target threshold. To find the upper bound, we find
       the smallest epsilon scaling factor less than the target threshold.
       If a lower bound is not found, it is set to the minimum of all
       candidate scaling factors divided by two. If an upper bound is not 
       found, it is set to the maximum of all candidate scaling factors 
       multiplied by two. 

       We then generate 5 evenly spaced values between the lower and upper
       bound, and use the middle three elements as the new scaling factor 
       candidates. 
    """
   
    logger.info('GETTING NEW SCALING FACTOR CANDIDATES GIVEN CURRENT:')
    logger.info(rw_hp_acceptance_rate_dict)

    num_chains = rw_hp_config_data['num_chains']
    chains_to_update = rw_hp_config_data['chains_to_update']
    chain_scaling_factor_type = rw_hp_config_data['chain_scaling_factor_type'] 

    if chains_to_update != ['all']:
        relevant_scaling_factor_idx = [index for index, value in enumerate(chains_to_update) if value == 1] #these correspond to the indices of the chains whose scaling factor is being tuned
    else:
        if chain_scaling_factor_type == 'uniform':
            relevant_scaling_factor_idx = [0]
        else:
            relevant_scaling_factor_idx = list(range(0,num_chains))

    curr_scaling_factor_candidates = list(rw_hp_acceptance_rate_dict.keys())
    curr_scaling_factor_candidates_rel_idx = [tuple(curr_scaling_factor_candidates[i][j] for j in relevant_scaling_factor_idx) for i in range(len(curr_scaling_factor_candidates))] 
    logger.info('SCALING FACTORS OF CHAINS BEING MODIFIED:')
    logger.info(curr_scaling_factor_candidates_rel_idx)

    upper_bound_scaling_factor = None
    lower_bound_scaling_factor = None 

    #find largest epsilon scaling factor (where largest defined in terms of sum) greater than target threshold 
    for key in sorted(curr_scaling_factor_candidates,key=lambda x: sum(x), reverse=True): 
        acceptance_rate = rw_hp_acceptance_rate_dict[key]
        if acceptance_rate > rw_hp_config_data['rw_tuning_acceptance_threshold']:
            key_relevant_idx = tuple(key[i] for i in relevant_scaling_factor_idx)
            lower_bound_scaling_factor = min(key_relevant_idx)
            break

    #find smallest epsilon scaling factor (where largest defined in terms of sum) less than target threshold 
    for key in sorted(curr_scaling_factor_candidates,key=lambda x: sum(x)): 
        acceptance_rate = rw_hp_acceptance_rate_dict[key]
        if acceptance_rate < rw_hp_config_data['rw_tuning_acceptance_threshold']:
            key_relevant_idx = tuple(key[i] for i in relevant_scaling_factor_idx)
            upper_bound_scaling_factor = max(key_relevant_idx)
            break

    curr_scaling_factor_candidates_rel_idx_flattened = list(itertools.chain(*curr_scaling_factor_candidates_rel_idx))
 
    if lower_bound_scaling_factor is None:
        lower_bound_scaling_factor = min(curr_scaling_factor_candidates_rel_idx_flattened)/2    
    if upper_bound_scaling_factor is None:
        upper_bound_scaling_factor = max(curr_scaling_factor_candidates_rel_idx_flattened)*2

    new_scaling_factor_candidates = list(np.linspace(lower_bound_scaling_factor, upper_bound_scaling_factor, 5))
    new_scaling_factor_candidates = new_scaling_factor_candidates[1:-1] #exclude first and last element
    return new_scaling_factor_candidates



def construct_grid_search_combinations(
    rw_hp_config_data: Mapping[str, Any], 
    scaling_factor_candidates: List[float]
):
    """Based on the scaling factor candidates, constructs the relevant 
       combination of hyperparameters to explore. 

       The hyperparameters are generated according a grid search. If 
       the scaling factor candidates are [x1,x2], we would   
       generate the combinations [(x1,x2),(x2,x1)] (if the 
       chain_scaling_factor_type = variable) or simply [(x1,x1), (x2,x2)] 
       (if the chain_scaling_factor_type = uniform).  
    """

    num_chains = rw_hp_config_data['num_chains']
    chains_to_update = rw_hp_config_data['chains_to_update']
    chain_scaling_factor_type = rw_hp_config_data['chain_scaling_factor_type'] 

    if chain_scaling_factor_type == 'variable':

        scaling_factor_candidates_all = [] 
        for i in range(0,num_chains):
            scaling_factor_candidates_all.append(scaling_factor_candidates)
        grid_search_combinations = list(itertools.product(*scaling_factor_candidates_all))

        if chains_to_update != ['all']: #chains_to_update is a binary vector of length num_chains
            for i in range(0,len(grid_search_combinations)):
                grid_search_combinations[i] = list(grid_search_combinations[i])
                for j in range(0,len(grid_search_combinations[i])):
                    if chains_to_update[j] == 0:
                        grid_search_combinations[i][j] = 1. 
                grid_search_combinations[i] = tuple(grid_search_combinations[i])

        grid_search_combinations = list(set(grid_search_combinations)) #remove duplicates              
        grid_search_combinations = sorted(grid_search_combinations, key=lambda x: sum(x))

        return grid_search_combinations
 
    elif chain_scaling_factor_type == 'uniform':

        if chains_to_update != ['all']: #chains_to_update is a binary vector of length num_chains, so only a subset of chains are to be updated 
            scaling_factor_candidates_all = []
            for i in range(0,len(scaling_factor_candidates)):
                scaling_factor_candidates_all.append(tuple([scaling_factor_candidates[i]]*num_chains))

            for i in range(0,len(scaling_factor_candidates_all)):
                scaling_factor_candidates_all[i] = list(scaling_factor_candidates_all[i])
                for j in range(0,len(scaling_factor_candidates_all[i])):
                    if chains_to_update[j] == 0:
                        scaling_factor_candidates_all[i][j] = 1. #keep this scaling factor constant  
                scaling_factor_candidates_all[i] = tuple(scaling_factor_candidates_all[i])
        else: #only single scaling factor is being applied to all chains
            scaling_factor_candidates_all = []
            for i in range(0,len(scaling_factor_candidates)):
                scaling_factor_candidates_all.append(tuple([scaling_factor_candidates[i]]))

        scaling_factor_candidates_all = sorted(scaling_factor_candidates_all, key=lambda x: sum(x))

        return scaling_factor_candidates_all 


def run_grid_search_multimer(
    grid_search_combinations: List[float], 
    state_history_dict: Optional[Mapping[Tuple[float, ...], int]], 
    source_str: str, 
    target_str: str, 
    initial_pred_path: str, 
    intrinsic_dim: int, 
    num_total_steps: int, 
    L: np.ndarray, 
    model: nn.Module, 
    config: mlc.ConfigDict, 
    feature_processor: feature_pipeline.FeaturePipeline, 
    feature_dict: FeatureDict, 
    processed_feature_dict: FeatureDict, 
    rw_hp_config_data: Mapping[str, Any], 
    output_dir: str, 
    args: argparse.Namespace
):
    """Runs a random walk for each set of hyperparameters in grid_search_combinations
       for num_total_steps. 
    
       We employ a simple heuristic to terminate the search process early if certain
       criteria or conditions are satisfied. 
    """

    rw_type = rw_hp_config_data['rw_type']
    cov_type = rw_hp_config_data['cov_type'] 

    upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
    lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)

    # To speed things up when searching for scaling factors, we initially evaluate the min(grid_search_combinations) 
    # and max(grid_search_combinations). Based on the results of the min(grid_search_combinations) and 
    # max(grid_search_combinations), we may terminate the search early. We only do this for grid_search_combinations
    # that have not been previously evaluated and consist of more than a single combination.

    if len(grid_search_combinations) == 1 or len(state_history_dict) > 0:

        for i,items in enumerate(grid_search_combinations): 

            rw_hp_output_dir = '%s/combo_num=%d/%s/%s' % (output_dir, i, source_str, target_str)
            rw_hp_dict = rw_helper_functions.populate_rw_hp_dict(rw_type, items, is_multimer=True)

            logger.info('EVALUATING RW HYPERPARAMETERS:')
            logger.info(rw_hp_dict)

            pdb_files = glob.glob('%s/**/*.pdb' % rw_hp_output_dir)
            if len(pdb_files) > 0: #restart
                logger.info('removing pdb files in %s' % rw_hp_output_dir)
                rw_helper_functions.remove_files(pdb_files)

            logger.info('BEGINNING RW FOR: %s' % rw_hp_output_dir)

            state_history, conformation_info = run_rw_multimer(initial_pred_path, intrinsic_dim, rw_type, rw_hp_dict, num_total_steps, L, cov_type, model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_output_dir, 'rw_grid_search', args, save_intrinsic_param=False, terminate_after_first_rejection=True)
            shutil.rmtree(rw_hp_output_dir, ignore_errors=True)

            if items not in state_history_dict:
                state_history_dict[items] = state_history
            else:
                state_history_dict[items].extend(state_history)

            acceptance_rate = round(sum(state_history_dict[items])/len(state_history_dict[items]),2)
            if args.early_stop_rw_hp_tuning:
                if acceptance_rate <= upper_bound_acceptance_threshold and acceptance_rate >= lower_bound_acceptance_threshold:
                    #mark all combinations except for current one with acceptance rate of '-1' so we ignore them
                    state_history_dict = rw_helper_functions.autopopulate_state_history_dict(state_history_dict, grid_search_combinations,
                                                                                             num_total_steps, optimal_combination=items, extrapolation_val=-1)
                    return state_history_dict

    else:

        # Precondition: grid_searching_combinations is sorted in ascending order by sum of all elements in each tuple
        min_max_combination = [grid_search_combinations[0], grid_search_combinations[-1]]
        grid_search_combinations_excluding_min_max = grid_search_combinations[1:-1]
        grid_search_combinations_reordered = min_max_combination
        grid_search_combinations_reordered.extend(grid_search_combinations_excluding_min_max)

        # If the acceptance rate of max(grid_search_combinations) >= ub_threshold, we set the acceptance rate of all 
        # other combinations to 1 (because decreasing scaling factor should generally increase acceptance rate). If the acceptance
        # rate of min(grid_search_combinations) <= lb_threshold, we set the acceptance rate of all other combinations 
        # to 0 (because increasing scaling factor should only decrease acceptance rate).
 
        for i,items in enumerate(grid_search_combinations_reordered): 

            rw_hp_output_dir = '%s/combo_num=%d/%s/%s' % (output_dir, i, source_str, target_str)
            rw_hp_dict = rw_helper_functions.populate_rw_hp_dict(rw_type, items, is_multimer=True)

            logger.info('EVALUATING RW HYPERPARAMETERS:')
            logger.info(rw_hp_dict)  

            pdb_files = glob.glob('%s/**/*.pdb' % rw_hp_output_dir)
            if len(pdb_files) > 0: #restart
                logger.info('removing pdb files in %s' % rw_hp_output_dir)
                rw_helper_functions.remove_files(pdb_files)

            logger.info('BEGINNING RW FOR: %s' % rw_hp_output_dir)

            state_history, conformation_info = run_rw_multimer(initial_pred_path, intrinsic_dim, rw_type, rw_hp_dict, num_total_steps, L, cov_type, model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_output_dir, 'rw_grid_search', args, save_intrinsic_param=False, terminate_after_first_rejection=True)
            shutil.rmtree(rw_hp_output_dir, ignore_errors=True)

            if items not in state_history_dict:
                state_history_dict[items] = state_history
            else:
                state_history_dict[items].extend(state_history)

            acceptance_rate = round(sum(state_history_dict[items])/len(state_history_dict[items]),2)
            if args.early_stop_rw_hp_tuning:
                if acceptance_rate <= upper_bound_acceptance_threshold and acceptance_rate >= lower_bound_acceptance_threshold:
                    #mark all combinations except for current one with acceptance rate of '-1' so we ignore them
                    state_history_dict = rw_helper_functions.autopopulate_state_history_dict(state_history_dict, grid_search_combinations,
                                                                                             num_total_steps, optimal_combination=items, extrapolation_val=-1)
                    return state_history_dict
                
            if i == 0: #min_combination
                if acceptance_rate < lower_bound_acceptance_threshold:
                    state_history_dict = rw_helper_functions.autopopulate_state_history_dict(state_history_dict, grid_search_combinations, 
                                                                                             num_total_steps, optimal_combination=None, extrapolation_val=acceptance_rate) 
                    return state_history_dict
                elif acceptance_rate >= lower_bound_acceptance_threshold and acceptance_rate <= upper_bound_acceptance_threshold:
                    min_combo_outside_acceptance_range = False
                else:
                    min_combo_outside_acceptance_range = True
            elif i == 1: #max_combination
                if acceptance_rate > upper_bound_acceptance_threshold:
                    #in general, if max_combo acceptance > ub_threshold, 
                    #then min_combo should as well. however, if the prediction 
                    #confidence is low, this can lead to more variability
                    #in the acceptance rate, so we check to see if min_combo
                    #is also outside the acceptance range before terminating
                    #the search early  
                    if min_combo_outside_acceptance_range: 
                        state_history_dict = rw_helper_functions.autopopulate_state_history_dict(state_history_dict, grid_search_combinations, 
                                                                                                 num_total_steps, optimal_combination=None, extrapolation_val=acceptance_rate) 
                        return state_history_dict
                
    return state_history_dict



def summarize_rw(
    model_name_source: str, 
    conformation_info: List[Tuple[Any,...]]
):
    """Prints out summary statistics of random walk. 
    """

    logger.info(asterisk_line)
    logger.info('RESULTS FOR MODEL: %s' % model_name_source)
    logger.debug(conformation_info)
    rmsd_all = np.array([conformation_info[i][1] for i in range(0,len(conformation_info))])
    iptm_score_all = np.array([conformation_info[i][3] for i in range(0,len(conformation_info))])
    max_rmsd = np.max(rmsd_all)
    mean_iptm_score = np.mean(iptm_score_all)
    max_iptm_score = np.max(iptm_score_all)
    logger.info('MAX RMSD: %.3f' % max_rmsd)
    logger.info('MEAN IPTM_PTM: %.3f' % mean_iptm_score)
    logger.info('MAX IPTM_PTM: %.3f' % max_iptm_score)
    logger.info(asterisk_line)



def run_rw_pipeline(args):

    if args.log_level.lower() == 'debug':
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    models_to_run = ['model_1_multimer_v3', 'model_2_multimer_v3', 'model_3_multimer_v3', 'model_4_multimer_v3', 'model_5_multimer_v3']
    jax_param_path_dict = {} 
    for m in models_to_run:
        jax_param_path_dict[m] = ''.join((args.jax_param_parent_path, '/params_', m, '.npz'))
    
    logger.info("MODEL LIST:")
    logger.info(jax_param_path_dict)

    output_dir = '%s/%s/%s/rw-%s/train-%s' % (args.output_dir_base, 'alternative_conformations-verbose', args.module_config, args.rw_hp_config, args.train_hp_config)
    l0_output_dir = '%s/%s' % (args.output_dir_base, 'alternative_conformations-verbose')
    l1_output_dir = '%s/%s/%s' % (args.output_dir_base, 'alternative_conformations-verbose', args.module_config)
    l2_output_dir = '%s/%s/%s/rw-%s' % (args.output_dir_base, 'alternative_conformations-verbose', args.module_config, args.rw_hp_config)
    
    output_dir = os.path.abspath(output_dir)
    l0_output_dir = os.path.abspath(l0_output_dir)
    l1_output_dir = os.path.abspath(l1_output_dir)
    l2_output_dir = os.path.abspath(l2_output_dir) 

    logger.info("OUTPUT DIRECTORY: %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)
    alignment_dir = args.alignment_dir
    msa_files = glob.glob('%s/*.a3m' % alignment_dir)
    if len(msa_files) == 0:
        file_id = os.listdir(alignment_dir)
        if len(file_id) > 1:
            raise ValueError("should only be a single directory under %s" % alignment_dir)
        else:
            file_id = file_id[0] #e.g 1xyz-1xyz
        alignment_dir_w_file_id = '%s/%s' % (alignment_dir, file_id)
        alignment_dir_wo_file_id = alignment_dir
    else:
        file_id = alignment_dir.split('/')[-1]
        file_id_wo_chain = file_id.split('_')[0]
        alignment_dir_w_file_id = alignment_dir
        alignment_dir_wo_file_id = alignment_dir[0:alignment_dir.rindex('/')]
    logger.info("alignment directory with file_id: %s" % alignment_dir_w_file_id)
    
    if args.fasta_file is None:
        pattern = "%s/*.fasta" % alignment_dir_w_file_id
        files = glob.glob(pattern, recursive=True)
        if len(files) == 1:
            fasta_file = files[0]
        else: 
            raise FileNotFoundError("Either >1 or 0 .fasta files found in alignment_dir -- should only be one")
    else:
        fasta_file = args.fasta_file

    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seqs = parse_fasta(fasta_data)
    logger.info("PROTEIN SEQUENCE:")
    logger.info(seqs)

    total_seq_len = sum(len(s) for s in seqs)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
 
    with open('./rw_multimer_config.json') as f:
        rw_config_data = json.load(f)

    module_config_data = rw_config_data['finetuning-method_SAID'][args.module_config]
    rw_hp_config_data = rw_config_data['hyperparameter']['rw'][args.rw_hp_config]
    train_hp_config_data = rw_config_data['hyperparameter']['train'][args.train_hp_config]
    intrinsic_dim = module_config_data['intrinsic_dim']

    pattern = "%s/features.pkl" % alignment_dir_w_file_id
    files = glob.glob(pattern, recursive=True)
    if len(files) == 1:
        features_output_path = files[0]
        logger.info('features.pkl path: %s' % features_output_path)
    else:
        features_output_path = ''

    if os.path.isfile(features_output_path):
        feature_dict = np.load(features_output_path, allow_pickle=True) 
    else:
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=4,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
        data_processor_monomer = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )
        data_processor_multimer = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor_monomer
        )
        feature_dict = data_processor_multimer.process_fasta(
            fasta_path=fasta_file, alignment_dir=alignment_dir_w_file_id,
        )

        features_output_path = os.path.join(alignment_dir_w_file_id, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)
        logger.info('SAVED %s' % features_output_path)

    logger.debug("FEATURE DICTIONARY:")
    logger.debug(feature_dict)
    num_chains = int(feature_dict['assembly_num_chains']) 
    module_config_data['num_chains'] = num_chains 
    rw_hp_config_data['num_chains'] = num_chains
    logger.info('NUM CHAINS: %d' % num_chains)

    chains_to_update = rw_hp_config_data['chains_to_update']
    chain_scaling_factor_type = rw_hp_config_data['chain_scaling_factor_type'] 

    logger.info("CHAINS TO UPDATE:")
    logger.info(chains_to_update)
    logger.info("CHAIN SCALING FACTOR TYPE:")
    logger.info(chain_scaling_factor_type)

    #perform checks on config file
    if chains_to_update != ['all']:
        if len(chains_to_update) != num_chains:
            raise ValueError("if chains_to_update not 'all', then must be a binary vector of length num_chains")
        for c in chains_to_update:
            if c not in [0,1]:
                raise ValueError("if chains_to_update not 'all', then must be a binary vector of length num_chains")
    if chain_scaling_factor_type not in ["uniform", "variable"]:
        raise ValueError("chain_scaling_factor_type must either be 'uniform' or 'variable'")
    if module_config_data['layer_to_update'] == ['all'] and chain_scaling_factor_type == 'variable':
        raise ValueError("layer_to_update cannot be 'all' when chain_scaling_factor_type is 'variable'. 'variable' is only allowed when a subset of specific layers with are being modified")

    if chains_to_update == ['all'] and chain_scaling_factor_type == 'uniform':
        use_chainmask = False
    else:
        use_chainmask = True 

    config_dict = {}
    for m in models_to_run:
        if use_chainmask:
            config_name = 'multimer-chainmask-%s' % m #model with chainmask  
        else:
            config_name = 'multimer-%s' % m #model without chainmask 
        config = model_config(config_name, long_sequence_inference=args.long_sequence_inference)
        config = update_config(config, seqs, num_chains, args)
        
        if total_seq_len > SEQ_LEN_EXTRAMSA_THRESHOLD:
            print('REDUCING MAX_EXTRA_MSA TO 512 BECAUSE TOTAL SEQUENCE LENGTH IS: %d' % total_seq_len)
            config.data.predict.max_extra_msa = 512

        if m not in config_dict:
            config_dict[m] = config
            if(args.trace_model):
                if(not config.data.predict.fixed_size):
                    raise ValueError(
                        "Tracing requires that fixed_size mode be enabled in the config"
                    )

    feature_processor = feature_pipeline.FeaturePipeline(config_dict[list(config_dict.keys())[0]].data)
    intrinsic_param_zero = np.zeros(intrinsic_dim)
 
    #####################################################
    logger.info(asterisk_line)

    initial_pred_output_dir = '%s/initial_pred' %  l0_output_dir

    if args.skip_initial_pred_phase:
        initial_pred_info_fname = '%s/initial_pred_info.pkl' % initial_pred_output_dir
        seed_fname = '%s/seed.txt' % initial_pred_output_dir
        if os.path.exists(initial_pred_info_fname) and os.path.exists(seed_fname):
            with open(initial_pred_info_fname, 'rb') as f:
                initial_pred_path_dict = pickle.load(f)
            random_seed = int(np.loadtxt(seed_fname))
            np.random.seed(random_seed)
            torch.manual_seed(random_seed + 1)
            logger.info('SKIPPING INITIAL PRED PHASE')
        else:
            raise FileNotFoundError('%s not found' % initial_pred_info_fname)

        #process features after updating seed 
        logger.debug('PROCESSING FEATURES')
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=True
        )
        processed_feature_dict = {
            k:torch.as_tensor(v, device=args.model_device)
            for k,v in processed_feature_dict.items()
        }
    else:
        logger.info('BEGINNING INITIAL PRED PHASE:') 
        t0 = time.perf_counter()

        #this uses the standard inference configuration with dropout for the initial prediction 
        model_dict = {} 
        for m in models_to_run:
            config_name = 'multimer-%s' % m #model without chainmask 
            config = model_config(config_name, long_sequence_inference=args.long_sequence_inference)
            config = update_config(config, seqs, num_chains, args) 
            model = load_model_w_intrinsic_param(config, 
                                                 module_config_data, 
                                                 args.model_device, 
                                                 None, 
                                                 jax_param_path_dict[m], 
                                                 intrinsic_param_zero, 
                                                 enable_dropout=True)
            if m not in model_dict:
                model_dict[m] = model

        logger.debug('PROCESSING FEATURES')
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=True
        )
        processed_feature_dict = {
            k:torch.as_tensor(v, device=args.model_device)
            for k,v in processed_feature_dict.items()
        }

        initial_pred_path_dict = {} 
        conformation_info_dict = {}
        for i in range(0,len(models_to_run)):
            model_name = models_to_run[i]
            curr_model_initial_pred_output_dir = '%s/%s' %  (initial_pred_output_dir, model_name)
            tag = 'initial_pred_model_%d' % (i+1)
            logger.info('RUNNING %s' % model_name)
            mean_plddt_initial, weighted_ptm_score_initial,  disordered_percentage_initial, _, _, _, initial_pred_path = eval_model(model_dict[model_name], config, intrinsic_param_zero, intrinsic_param_zero, [0]*num_chains, feature_processor, feature_dict, processed_feature_dict, tag, curr_model_initial_pred_output_dir, 'initial', args)   
            logger.info('pLDDT: %.3f, IPTM_PTM SCORE: %.3f, disordered percentage: %.3f' % (mean_plddt_initial, weighted_ptm_score_initial, disordered_percentage_initial)) 
            conformation_info_dict[model_name] = (initial_pred_path, mean_plddt_initial, weighted_ptm_score_initial, disordered_percentage_initial) 
            initial_pred_path_dict[model_name] = initial_pred_path 

        if args.write_summary_dir:
            summary_output_dir = '%s/%s/initial_pred' % (args.output_dir_base, 'alternative_conformations-summary')
            os.makedirs(summary_output_dir, exist_ok=True)
            rw_helper_functions.remove_files_in_dir(summary_output_dir)
            shutil.copytree(initial_pred_output_dir, summary_output_dir, dirs_exist_ok=True)

        run_time = time.perf_counter() - t0
        timing_dict = {'initial_pred': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'inital_pred')

        rw_helper_functions.dump_pkl(initial_pred_path_dict, 'initial_pred_info', initial_pred_output_dir)
        rw_helper_functions.dump_pkl(conformation_info_dict, 'conformation_info', initial_pred_output_dir)

        seed_fname = '%s/seed.txt' % initial_pred_output_dir
        np.savetxt(seed_fname, [random_seed], fmt='%d')
       
    aligned_models_info = get_aligned_models_info(models_to_run, initial_pred_path_dict, file_id, args)
 
    #####################################################
    logger.info(asterisk_line)

    #this uses an inference configuration with or without the chainmask (as specified by user) and without dropout (unless specified otherwise by user) 
    model_dict = {} 
    for m in models_to_run: 
        model = load_model_w_intrinsic_param(config_dict[m], 
                                             module_config_data, 
                                             args.model_device, 
                                             None, 
                                             jax_param_path_dict[m], 
                                             intrinsic_param_zero, 
                                             enable_dropout=args.enable_dropout)
        if m not in model_dict:
            model_dict[m] = model

    if not(args.skip_gd_phase): 
        logger.info('BEGINNING GD PHASE:') 
        t0 = time.perf_counter()
        for i in range(0,len(aligned_models_info)): 
            finetune_wrapper(aligned_models_info[i], jax_param_path_dict, 
                             alignment_dir_wo_file_id, output_dir, args)
        run_time = time.perf_counter() - t0
        timing_dict = {'gradient_descent': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'gradient_descent')
    

    #####################################################
    logger.info(asterisk_line)

    if rw_hp_config_data['cov_type'] == 'full':
        logger.info('GENERATING RANDOM CORRELATION MATRIX')
        if args.use_local_context_manager:
            with local_np_seed(random_seed):
                random_corr = gen_randcorr_sap.randcorr(intrinsic_dim)
        else:
            random_corr = gen_randcorr_sap.randcorr(intrinsic_dim)
    
    rw_hp_dict = {}  
    rw_hp_parent_dir = '%s/rw_hp_tuning' % output_dir
    rw_hp_acceptance_rate_dict = {}  
    rw_hp_acceptance_rate_fname = '%s/rw_hp_acceptance_rate_info.pkl' % rw_hp_parent_dir

    skip_auto_calc = False
    if os.path.exists(rw_hp_acceptance_rate_fname):
        logger.info('LOADING %s' % rw_hp_acceptance_rate_fname)
        with open(rw_hp_acceptance_rate_fname, 'rb') as f:
            rw_hp_acceptance_rate_dict = pickle.load(f)
        logger.info(rw_hp_acceptance_rate_dict)
        rw_hp_dict = rw_helper_functions.get_optimal_hp(rw_hp_acceptance_rate_dict, rw_hp_config_data, is_multimer=True)
        if rw_hp_dict != {}:
            skip_auto_calc = True

    if not(skip_auto_calc):

        logger.info('BEGINNING RW HYPERPARAMETER TUNING PHASE')
        t0 = time.perf_counter()

        upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
        lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)

        scaling_factor_candidates = train_hp_config_data['initial_scaling_factor_candidate']
        scaling_factor_candidates = list(map(float, scaling_factor_candidates))  
        
        while rw_hp_dict == {}:

            state_history_dict = {} 
            grid_search_combinations = construct_grid_search_combinations(rw_hp_config_data, scaling_factor_candidates)
            logger.info('INITIAL GRID SEARCH PARAMETERS:')
            logger.info(grid_search_combinations)

            for i in range(0,len(aligned_models_info)):
                model_name_source = aligned_models_info[i][0] #this is model being used for training 
                model_name_target = aligned_models_info[i][1] 
                source_pdb_path = aligned_models_info[i][2]
                logger.info('ON MODEL: %s' % model_name_source)
                logger.info(aligned_models_info[i])
                source_str = 'source=%s' % model_name_source #corresponds to model_x_multimer_v3 
                target_str = 'target=%s' % model_name_target
                fine_tuning_save_dir = '%s/training/%s/%s' % (output_dir, source_str, target_str)
                latest_version_num = rw_helper_functions.get_latest_version_num(fine_tuning_save_dir)
                model_train_out_dir = '%s/version_%d' % (fine_tuning_save_dir, latest_version_num)
                sigma = rw_helper_functions.get_sigma(intrinsic_dim, model_train_out_dir) 
                L = rw_helper_functions.get_cholesky(rw_hp_config_data['cov_type'], sigma, random_corr)
                state_history_dict = run_grid_search_multimer(grid_search_combinations, state_history_dict, source_str, target_str, source_pdb_path, intrinsic_dim, args.num_rw_hp_tuning_steps_per_round, L, model_dict[model_name_source], config_dict[model_name_source], feature_processor, feature_dict, processed_feature_dict, rw_hp_config_data, rw_hp_parent_dir, args)
                rw_hp_acceptance_rate_dict, grid_search_combinations, completion_status = rw_helper_functions.get_rw_hp_tuning_info(state_history_dict, rw_hp_acceptance_rate_dict, grid_search_combinations, rw_hp_config_data, i, args)
                if completion_status == 1:
                    break 
                      
            rw_hp_dict = rw_helper_functions.get_optimal_hp(rw_hp_acceptance_rate_dict, rw_hp_config_data, is_multimer=True)
            if rw_hp_dict == {}:
                logger.info('NO SCALING FACTOR CANDIDATES FOUND THAT MATCHED ACCEPTANCE CRITERIA')
                scaling_factor_candidates = get_new_scaling_factor_candidates(rw_hp_acceptance_rate_dict, rw_hp_config_data)
                logger.info('HYPERPARAMETER TUNING WITH NEW SCALING FACTOR CANDIDATES')
                logger.info(scaling_factor_candidates)
            else:
                rw_helper_functions.dump_pkl(rw_hp_acceptance_rate_dict, 'rw_hp_acceptance_rate_info', rw_hp_parent_dir)            
                logger.info(rw_hp_acceptance_rate_dict)

        run_time = time.perf_counter() - t0
        timing_dict = {'hp_tuning': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'hp_tuning')
                                 
    #####################################################
    logger.info(asterisk_line)

    logger.info('BEGINNING RW PHASE')
    logger.info('HYPERPARAMETERS BEING USED:')
    logger.info(rw_hp_dict)

    for i in range(0,len(aligned_models_info)):
        model_name_source = aligned_models_info[i][0] #this is model being used for training 
        model_name_target = aligned_models_info[i][1] 
        source_pdb_path = aligned_models_info[i][2]
        logger.info('ON MODEL: %s' % model_name_source)
        t0 = time.perf_counter()

        conformation_info_dict = {} #maps source_target to (pdb_path,plddt,disordered_percentage,rmsd) 
        source_str = 'source=%s' % model_name_source #corresponds to model_x_multimer_v3 
        target_str = 'target=%s' % model_name_target
        fine_tuning_save_dir = '%s/training/%s/%s' % (output_dir, source_str, target_str)
        latest_version_num = rw_helper_functions.get_latest_version_num(fine_tuning_save_dir)

        model_train_out_dir = '%s/version_%d' % (fine_tuning_save_dir, latest_version_num)
        sigma = rw_helper_functions.get_sigma(intrinsic_dim, model_train_out_dir)
        L = rw_helper_functions.get_cholesky(rw_hp_config_data['cov_type'], sigma, random_corr)
 
        rw_output_dir = '%s/rw_output/%s/%s' % (output_dir,source_str,target_str)
        #removes pdb files if iteration should be overwritten or restarted
        should_run_rw = rw_helper_functions.overwrite_or_restart_incomplete_iterations(rw_output_dir, args)
        if not(should_run_rw):
            continue 

        logger.info('BEGINNING RW FOR: %s' % rw_output_dir)

        if args.use_local_context_manager:
            with local_np_seed(random_seed):
                state_history, conformation_info = run_rw_multimer(source_pdb_path, intrinsic_dim, rw_hp_config_data['rw_type'], rw_hp_dict, args.num_rw_steps, L, rw_hp_config_data['cov_type'], model_dict[model_name_source], config_dict[model_name_source], feature_processor, feature_dict, processed_feature_dict, rw_output_dir, 'rw', args, save_intrinsic_param=False, terminate_after_first_rejection=False)
        else:
            state_history, conformation_info = run_rw_multimer(source_pdb_path, intrinsic_dim, rw_hp_config_data['rw_type'], rw_hp_dict, args.num_rw_steps, L, rw_hp_config_data['cov_type'], model_dict[model_name_source], config_dict[model_name_source], feature_processor, feature_dict, processed_feature_dict, rw_output_dir, 'rw', args, save_intrinsic_param=False, terminate_after_first_rejection=False)

        key = '%s-%s' % (model_name_source, model_name_target)
        conformation_info_dict[key] = conformation_info

        acceptance_rate = sum(state_history)/len(state_history)
        logger.info('ACCEPTANCE RATE: %.3f' % acceptance_rate)

        rw_helper_functions.dump_pkl(conformation_info_dict, 'conformation_info', rw_output_dir)
        summarize_rw(model_name_source, conformation_info)

        inference_key = 'inference_%d' % i
        run_time = time.perf_counter() - t0
        timing_dict = {inference_key: run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, inference_key)
 
    if args.write_summary_dir:
        rw_output_parent_dir = '%s/rw_output' % output_dir
        summary_output_dir = '%s/%s/rw_output' % (args.output_dir_base, 'alternative_conformations-summary')
        os.makedirs(summary_output_dir, exist_ok=True)
        rw_helper_functions.remove_files_in_dir(summary_output_dir)
        shutil.copytree(rw_output_parent_dir, summary_output_dir, dirs_exist_ok=True)


           
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
        "--alignment_dir", type=str, default=None,
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
        "--num_rw_steps", type=int, default=100
    )
    parser.add_argument(
        "--num_rw_hp_tuning_steps_per_round", type=int, default=10
    )
    parser.add_argument(
        "--num_rw_hp_tuning_rounds_total", type=int, default=2
    )
    parser.add_argument(
        "--early_stop_rw_hp_tuning", action="store_true", default=False,
    )
    parser.add_argument(
        "--num_training_conformations", type=int, default=5
    )
    parser.add_argument(
        "--save_training_conformations", action="store_true", default=False
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
        "--relax_conformation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=1,
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
        "--module_config", type=str, default=None,
        help=(
            "module_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--rw_hp_config", type=str, default=None,
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--train_hp_config", type=str, default=None,
        help=(
            "train_hp_config_x where x is a number"
        )
    )
    parser.add_argument(
    "--use_local_context_manager", action="store_true", default=False,
    help=(
        """whether to use local context manager
         when generating proposals. this means 
         that the same set of intrinsic_param
         will be produced within that context
         block."""
        )    
    )  
    parser.add_argument(
        "--skip_initial_pred_phase", action="store_true", default=False
    )    
    parser.add_argument(
        "--skip_gd_phase", action="store_true", default=False
    )    
    parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )
    parser.add_argument(
        "--write_summary_dir", type=bool, default=True
    )
    parser.add_argument(
        "--recycle_wo_early_stopping", action="store_true", default=False
    )
    parser.add_argument(
        "--max_recycling_iters", type=int, default=19
    )
    parser.add_argument(
        "--enable_dropout", action="store_true", default=False
    )
    parser.add_argument(
        "--mean_plddt_threshold", type=int, default=60
    )
    parser.add_argument(
        "--disordered_percentage_threshold", type=int, default=80
    )
    parser.add_argument(
        "--log_level", type=str, default='INFO'
    )

    add_data_args(parser)
    args = parser.parse_args()

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    run_rw_pipeline(args)

