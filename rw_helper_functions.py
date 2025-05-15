import argparse
import logging
import math
import numpy as np
import pandas as pd
import os
import shutil
import json
import re
import glob  
import sys
import itertools
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

from custom_openfold_utils.pdb_utils import convert_pdb_to_mmcif, get_bfactor


logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'

def write_timings(timing_dict, output_dir_base, timing_fname):
    output_dir = '%s/timings' % output_dir_base
    os.makedirs(output_dir, exist_ok=True)
    output_file = '%s/%s.json' % (output_dir, timing_fname) 
    with open(output_file, "w") as f:
        json.dump(timing_dict, f)


def dump_pkl(data, fname, output_dir):
    output_path = '%s/%s.pkl' % (output_dir, fname)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def remove_files_in_dir(path):
    file_list = sorted(glob.glob('%s/*' % path))
    for f in file_list:
        logger.info('removing old file: %s' % f)
        os.remove(f)


def remove_files(file_list):
    for f in file_list:
        logger.info('removing old file: %s' % f)
        os.remove(f)


def remove_trailing_zeros(number):
    number_str = str(number)
    number_str = number_str.rstrip('0').rstrip('.') if '.' in number_str else number_str
    return(number_str)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def get_latest_version_num(fine_tuning_save_dir):
    latest_version_num = -1
    if os.path.isdir(fine_tuning_save_dir):
        items = os.listdir(fine_tuning_save_dir)
        sub_dir = [item for item in items if os.path.isdir(os.path.join(fine_tuning_save_dir, item))]
        version_num_list = [] 
        for d in sub_dir:
            if 'version' in d:
                version_num = int(d.split('_')[1])
                version_num_list.append(version_num)
        latest_version_num = max(version_num_list)
    return latest_version_num

def get_af_ncterm_disordered_residues_idx(pdb_path: str):
    
    #plddt_threshold sourced from https://onlinelibrary.wiley.com/doi/10.1002/pro.4466

    cif_path = convert_pdb_to_mmcif(pdb_path, './cif_temp')
    plddt_scores, mean_plddt = get_bfactor(cif_path)
    os.remove(cif_path)

    af_disordered = 1-plddt_scores/100
    af_disordered = af_disordered >= .31
     
    af_ncterm_disordered_idx = []

    #n-terminal disordered domain 
    start_idx = 0
    if af_disordered[start_idx]:
        af_ncterm_disordered_idx.append(start_idx)
        end_idx = start_idx+1
        if end_idx < len(af_disordered):
            while af_disordered[end_idx]:
                af_ncterm_disordered_idx.append(end_idx)
                end_idx += 1
                if end_idx >= len(af_disordered):
                    break  

    #c-terminal disordered domain 
    start_idx = len(af_disordered)-1
    if af_disordered[start_idx]:
        af_ncterm_disordered_idx.append(start_idx)
        end_idx = start_idx-1
        if end_idx >= 0:
            while af_disordered[end_idx]:
                af_ncterm_disordered_idx.append(end_idx)
                end_idx -= 1
                if end_idx < 0:
                    break 

    logger.info("AF N/C TERM DISORDERED RESIDUES IDX:")
    logger.info(af_ncterm_disordered_idx)
 

    return af_ncterm_disordered_idx



def calc_disordered_percentage(pdb_path): 
    dssp_tuple = dssp_dict_from_pdb_file(pdb_path)
    dssp_dict = dssp_tuple[0]
    ss_all = [] 
    for key in dssp_dict:
        ss = dssp_dict[key][1]
        ss_all.append(ss)

    logger.info("SECONDARY STRUCTURE PROFILE:")
    logger.info(ss_all)

    disordered_num = 0 
    for s in ss_all:
        if s in ['-','T','S']:
            disordered_num += 1
    disordered_percentage = (disordered_num/len(ss_all))*100 
    return disordered_percentage


def accept_criteria(mean_plddt, disordered_percentage, mean_plddt_threshold, disordered_percentage_threshold):
    if (mean_plddt >= mean_plddt_threshold) and (disordered_percentage <= disordered_percentage_threshold):
        return 1
    else:
        return 0


def get_intrinsic_param_matrix(model_train_out_dir):

    out_fname = '%s/intrinsic_param_data.csv' % model_train_out_dir
    if os.path.isfile(out_fname):
        out = pd.read_csv(out_fname)
        return out 
        
    metrics_df = pd.read_csv('%s/metrics.csv' % model_train_out_dir)
    drmsd_ca_all = list(metrics_df['train/drmsd_ca'].dropna())
    lddt_ca_all = list(metrics_df['train/lddt_ca'].dropna())
    step_all = list(set(list(metrics_df['step'].dropna())))

    dim_size = 0 
    all_intrinsic_param = [] 
    all_filepaths = ['%s/step_%d.pt' % (model_train_out_dir,s) for s in step_all]
    for i,f in enumerate(all_filepaths):
        d = torch.load(f)
        intrinsic_param = d['intrinsic_parameter'].tolist()
        dim_size = len(intrinsic_param)
        drmsd_ca = drmsd_ca_all[i]
        lddt_ca = lddt_ca_all[i]
        step = step_all[i]
        intrinsic_param.append(drmsd_ca)
        intrinsic_param.append(step)
        all_intrinsic_param.append(intrinsic_param)

    cols = [('v%d' % i) for i in range(0,dim_size)]
    cols.append('drmsd_ca')
    cols.append('step_num')

    out = pd.DataFrame(all_intrinsic_param, columns = cols)
    out.to_csv(out_fname, index=False)
    return out  

def sample_from_normal(data, num_samples):
    mean = np.mean(data)
    std = np.std(data)
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    return samples 

#https://stats.stackexchange.com/questions/26185/estimation-of-white-noise-parameters-in-gaussian-random-walk-model
def get_sigma(intrinsic_dim, model_train_out_dir):
    intrinsic_param_matrix = get_intrinsic_param_matrix(model_train_out_dir)
    intrinsic_param_matrix = intrinsic_param_matrix.iloc[:,0:intrinsic_dim].to_numpy()
    intrinsic_param_diff = np.diff(intrinsic_param_matrix, axis=0)
    sigma = np.std(intrinsic_param_diff, axis=0)
    return sigma 

#https://stats.stackexchange.com/questions/62850/obtaining-covariance-matrix-from-correlation-matrix
def get_random_cov(sigma, random_corr):
    random_cov = (sigma[:, np.newaxis]*random_corr)*sigma
    return random_cov

def get_cholesky(cov_type, sigma, random_corr):
    if cov_type == 'full':
        random_cov = get_random_cov(sigma, random_corr)
        logger.info('calculating cholesky')
        L = np.linalg.cholesky(random_cov)
    else:
        L = None 
    return L 


def autopopulate_state_history_dict(state_history_dict, grid_search_combinations, num_total_steps, optimal_combination=None, extrapolation_val=-1):
    #this is called to make early termination work 
    for i,items in enumerate(grid_search_combinations):
        if items != optimal_combination:
            state_history_dict[items] = [extrapolation_val]*num_total_steps 
    return state_history_dict
         

def populate_rw_hp_dict(rw_type, items, is_multimer):
    rw_hp_dict = {} 
    if is_multimer:
        rw_hp_dict['epsilon_scaling_factor'] = list(items)
    else:
        if rw_type == 'vanila':
            rw_hp_dict['epsilon_scaling_factor'] = items[0]
        elif rw_type == 'discrete_ou':
            rw_hp_dict['epsilon_scaling_factor'] = items[0]
            rw_hp_dict['alpha'] = items[1]
        elif rw_type == 'rw_w_momentum':
            rw_hp_dict['epsilon_scaling_factor'] = items[0]
            rw_hp_dict['gamma'] = items[1]
    return rw_hp_dict



def get_optimal_hp(
    hp_acceptance_rate_dict: Mapping[Tuple[float, ...], float],
    rw_hp_config_data: Mapping[str, Any],
    is_multimer: bool = False
):
    if is_multimer:
        f = lambda x: sum(x)
    else:
        f = lambda x: x[0] #sort by eps scaling factor (eps scaling factor is always first element in zipped list) 

    logger.info('GETTING OPTIMAL HP GIVEN CURRENT:')
    logger.info(hp_acceptance_rate_dict)

    upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
    lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)
    
    acceptance_rate_delta_dict = {} 
    for key in sorted(hp_acceptance_rate_dict.keys(),key=f): 
            acceptance_rate = hp_acceptance_rate_dict[key]
            acceptance_rate_delta_dict[key] = abs(acceptance_rate - rw_hp_config_data['rw_tuning_acceptance_threshold'])

    #get hyperparameters whose acceptance rate is closest to acceptance_threshold (and at least as large as the acceptance threshold) 
    optimal_hyperparameters = None
    min_delta = 100
    for key in sorted(acceptance_rate_delta_dict.keys(),key=f):  
        acceptance_rate_delta = acceptance_rate_delta_dict[key]
        acceptance_rate = hp_acceptance_rate_dict[key]
        if acceptance_rate > upper_bound_acceptance_threshold or acceptance_rate < lower_bound_acceptance_threshold:
            continue 
        elif acceptance_rate_delta < min_delta:
            min_delta = acceptance_rate_delta
            optimal_hyperparameters = key

    if optimal_hyperparameters is not None:
        rw_hp_dict = populate_rw_hp_dict(rw_hp_config_data['rw_type'], optimal_hyperparameters, is_multimer)
    else:
        rw_hp_dict = {} 

    return rw_hp_dict


def get_rw_hp_tuning_info(
    state_history_dict: Optional[Mapping[Tuple[float, ...], int]], 
    hp_acceptance_rate_dict: Mapping[Tuple[float, ...], float], 
    grid_search_combinations: List[float],
    rw_hp_config_data: Mapping[str, Any],
    round_num: int,  
    args: argparse.Namespace
):

    upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
    lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)

    for key in state_history_dict: 
        logger.info('FOR RW HYPERPARAMETER COMBINATION:')
        logger.info(key)
        logger.info('STATE HISTORY:')
        logger.info(state_history_dict[key])
        cumm_acceptance_rate = sum(state_history_dict[key])/len(state_history_dict[key])
        logger.info('ACCEPTANCE RATE= %.3f' % cumm_acceptance_rate)
        hp_acceptance_rate_dict[key] = cumm_acceptance_rate

        if cumm_acceptance_rate > upper_bound_acceptance_threshold or cumm_acceptance_rate < lower_bound_acceptance_threshold:
            if key in grid_search_combinations:
                logger.info('REMOVING RW HYPERPARAMETER COMBINATION WITH ACCEPTANCE RATE %.2f' % cumm_acceptance_rate)
                logger.info(key)
                grid_search_combinations.remove(key)

    logger.info('CURRENT GRID SEARCH PARAMETERS:')
    logger.info(grid_search_combinations)

    completion_status = 0 

    if len(grid_search_combinations) == 0:
        logger.info('ALL CURRENT HYPERPARAMETER COMBINATIONS HAVE BEEN ELIMINATED')
        completion_status = 1
    elif len(grid_search_combinations) == 1:
        logger.info('ONLY SINGLE HYPERPARAMETER COMBINATION  EXISTS:')
        completion_status = 1
    elif len(grid_search_combinations) > 1:
        completed = False
        if (round_num+1) >= args.num_rw_hp_tuning_rounds_total:
            completed = True
        if completed:
            logger.info('ALL HYPERPARAMETER COMBINATIONS HAVE BEEN RUN THE SPECIFIED NUMBER OF ROUNDS (%d)' % args.num_rw_hp_tuning_rounds_total)
            completion_status = 1 

    return hp_acceptance_rate_dict, grid_search_combinations, completion_status


def overwrite_or_restart_incomplete_iterations(rw_output_dir, args):

    should_run_rw = True 

    pdb_files = glob.glob('%s/**/*.pdb' % rw_output_dir)
    if len(pdb_files) >= args.num_rw_steps:
        if args.overwrite_pred:
            logger.info('removing pdb files in %s' % rw_output_dir)
            remove_files(pdb_files)
        else:
            logger.info('SKIPPING RW FOR: %s --%d files already exist--' % (rw_output_dir, len(pdb_files)))       
            should_run_rw = False 
    elif len(pdb_files) > 0: #incomplete job
        logger.info('removing pdb files in %s' % rw_output_dir)
        remove_files(pdb_files)

    return should_run_rw 

