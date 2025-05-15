import logging
import os 
import io 
import subprocess 
import numpy as np
import pandas as pd 
import string
from io import StringIO
from pathlib import Path
import shutil 
import sys

from typing import List

from custom_openfold_utils.seq_utils import align_seqs, get_residues_idx_in_seq2_not_seq1, get_residues_idx_in_seq1_and_seq2 
from custom_openfold_utils.pdb_utils import get_ca_coords_dict, get_ca_coords_matrix, get_af_disordered_domains, get_af_disordered_residues, get_pdb_disordered_residues, get_pdb_path_seq 

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

def renumber_keys(residue_idx_ca_coords_dict, source):
    offset = min(residue_idx_ca_coords_dict.keys())
    if source == 'af':
        return {i: v for i, v in enumerate(residue_idx_ca_coords_dict.values())}, offset
    return residue_idx_ca_coords_dict, offset
    
def get_rmsf(pdb1_seq, pdb2_seq, pdb1_path, pdb2_path, pdb1_source, pdb2_source, pdb1_include_idx, pdb2_include_idx, pdb1_exclude_idx=[], pdb2_exclude_idx=[]):

    pdb1_seq_aligned, pdb2_seq_aligned, pdb1_seq_aligned_to_original_idx_mapping, pdb2_seq_aligned_to_original_idx_mapping = align_seqs(pdb1_seq, pdb2_seq)

    pdb1_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb1_path)
    pdb2_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb2_path)

    #because residues from af can be chopped off, renumber starting from 0, pdb files are assumed to be cleaned so they start from 1  
    pdb1_residue_idx_ca_coords_dict, pdb1_offset = renumber_keys(pdb1_residue_idx_ca_coords_dict, pdb1_source)
    pdb2_residue_idx_ca_coords_dict, pdb2_offset = renumber_keys(pdb2_residue_idx_ca_coords_dict, pdb2_source)

    pdb1_ca_pos_aligned = []
    pdb2_ca_pos_aligned = [] 

    out = []     

    for i in range(0,len(pdb1_seq_aligned)):
        if (pdb1_seq_aligned[i] == pdb2_seq_aligned[i]) and (pdb1_seq_aligned[i] != '-'):
            pdb1_seq_original_idx = pdb1_seq_aligned_to_original_idx_mapping[i]
            pdb2_seq_original_idx = pdb2_seq_aligned_to_original_idx_mapping[i]
            valid_idx = (pdb1_seq_original_idx in pdb1_include_idx) and (pdb2_seq_original_idx in pdb2_include_idx) and (pdb1_seq_original_idx not in pdb1_exclude_idx) and (pdb2_seq_original_idx not in pdb2_exclude_idx)
            if valid_idx:
                #print("aligned idx for pdb1: %d, seq idx: %d:" % (i, pdb1_seq_original_idx))
                #print("aligned idx for pdb2: %d, seq idx: %d:" % (i, pdb2_seq_original_idx))
                residue_exists_in_both_structures_bool = pdb1_seq_original_idx in pdb1_residue_idx_ca_coords_dict and pdb2_seq_original_idx in pdb2_residue_idx_ca_coords_dict
                if residue_exists_in_both_structures_bool:
                    pdb1_ca_pos = list(pdb1_residue_idx_ca_coords_dict[pdb1_seq_original_idx][0:3])
                    pdb2_ca_pos = list(pdb2_residue_idx_ca_coords_dict[pdb2_seq_original_idx][0:3])
                    curr_ca_rmsf = np.linalg.norm(np.array(pdb1_ca_pos) - np.array(pdb2_ca_pos))
                    out.append([pdb1_seq_original_idx+1,pdb2_seq_original_idx+1,pdb1_seq_original_idx+1+pdb1_offset,pdb2_seq_original_idx+1+pdb2_offset,pdb1_seq_aligned[i],pdb2_seq_aligned[i],curr_ca_rmsf])
 
    return out


 
def get_residue_idx_below_rmsf_threshold(pdb1_seq, pdb2_seq, pdb1_path, pdb2_path, pdb1_source, pdb2_source, pdb1_include_idx, pdb2_include_idx, pdb1_exclude_idx, pdb2_exclude_idx, rmsf_threshold=1.0):

    pdb1_seq_aligned, pdb2_seq_aligned, pdb1_seq_aligned_to_original_idx_mapping, pdb2_seq_aligned_to_original_idx_mapping = align_seqs(pdb1_seq, pdb2_seq)

    pdb1_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb1_path)
    pdb2_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb2_path)

    #because residues from af can be chopped off, renumber starting from 0, pdb files are assumed to be cleaned so they start from 1 
    pdb1_residue_idx_ca_coords_dict, pdb1_offset = renumber_keys(pdb1_residue_idx_ca_coords_dict, pdb1_source)
    pdb2_residue_idx_ca_coords_dict, pdb2_offset = renumber_keys(pdb2_residue_idx_ca_coords_dict, pdb2_source)

    pdb1_ca_pos_aligned = []
    pdb2_ca_pos_aligned = [] 

    pdb1_residue_idx_below_threshold = []
    pdb2_residue_idx_below_threshold = [] 

    for i in range(0,len(pdb1_seq_aligned)):
        if (pdb1_seq_aligned[i] == pdb2_seq_aligned[i]) and (pdb1_seq_aligned[i] != '-'):
            pdb1_seq_original_idx = pdb1_seq_aligned_to_original_idx_mapping[i]
            pdb2_seq_original_idx = pdb2_seq_aligned_to_original_idx_mapping[i]
            valid_idx = (pdb1_seq_original_idx in pdb1_include_idx) and (pdb2_seq_original_idx in pdb2_include_idx) and (pdb1_seq_original_idx not in pdb1_exclude_idx) and (pdb2_seq_original_idx not in pdb2_exclude_idx)
            if valid_idx:
                ####we want to check if the residue exists in any pdb derived structure - it could be missing (either due to cleaning process or missing atom). 
                ###for af derived structure, the residue should exist, so we don't check (this allows for more flexibility because its possible we chop off residues from af derived structure, which screws up with numbering)
                if pdb1_source == 'pdb' and pdb2_source == 'pdb': #check both
                    residue_exists_in_both_structures_bool = pdb1_seq_original_idx in pdb1_residue_idx_ca_coords_dict and pdb2_seq_original_idx in pdb2_residue_idx_ca_coords_dict
                elif pdb1_source == 'pdb' and pdb2_source == 'af': #check pdb1
                    residue_exists_in_both_structures_bool = pdb1_seq_original_idx in pdb1_residue_idx_ca_coords_dict
                elif pdb1_source == 'af' and pdb2_source == 'pdb': #check pdb2
                    residue_exists_in_both_structures_bool = pdb2_seq_original_idx in pdb2_residue_idx_ca_coords_dict 
                elif pdb1_source == 'af' and pdb2_source == 'af': #check neither 
                    residue_exists_in_both_structures_bool = True 
                if residue_exists_in_both_structures_bool:
                    pdb1_ca_pos = list(pdb1_residue_idx_ca_coords_dict[pdb1_seq_original_idx][0:3])
                    pdb2_ca_pos = list(pdb2_residue_idx_ca_coords_dict[pdb2_seq_original_idx][0:3])
                    curr_ca_rmsf = np.linalg.norm(np.array(pdb1_ca_pos) - np.array(pdb2_ca_pos))
                    if curr_ca_rmsf < rmsf_threshold:
                        pdb1_residue_idx_below_threshold.append(pdb1_seq_original_idx)
                        pdb2_residue_idx_below_threshold.append(pdb2_seq_original_idx)
                    pdb1_ca_pos_aligned.append(pdb1_ca_pos)
                    pdb2_ca_pos_aligned.append(pdb2_ca_pos)
                else:
                    pdb1_residue_idx_below_threshold.append(pdb1_seq_original_idx)
                    pdb2_residue_idx_below_threshold.append(pdb2_seq_original_idx)

                
    #pdb1_ca_pos_aligned = np.array(pdb1_ca_pos_aligned)
    #pdb2_ca_pos_aligned = np.array(pdb2_ca_pos_aligned)
    #ca_rmsf = np.linalg.norm(pdb1_ca_pos_aligned-pdb2_ca_pos_aligned, axis=1)
    #print(ca_rmsf)
    #print(pdb1_residue_idx_below_threshold)
    #print(pdb2_residue_idx_below_threshold)

    return pdb1_residue_idx_below_threshold, pdb2_residue_idx_below_threshold

def get_residues_ignore_idx_between_pdb_conformations(conformation1_pdb_path, conformation2_pdb_path, conformation1_af_initial_pred_path, conformation2_af_initial_pred_path):

    """Ignore residues based on:
        1. disorder
        2. common residues
        3. rmsf threshold

        conformation1/2_af_initial_pred_path is to determine disordered residues
        and corresponds to a standard, vanila AF prediction 

    """  

    conformation1_pdb_seq = get_pdb_path_seq(conformation1_pdb_path, None)
    conformation2_pdb_seq = get_pdb_path_seq(conformation2_pdb_path, None)
    conformation1_af_seq = get_pdb_path_seq(conformation1_af_initial_pred_path, None)
    conformation2_af_seq = get_pdb_path_seq(conformation2_af_initial_pred_path, None)

    try: 
        conformation1_af_disordered_domains_idx, _ = get_af_disordered_domains(conformation1_af_initial_pred_path)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % conformation1_af_initial_pred_path) 
        print(e)
        print('SKIPING THIS INPUT...')
        return None, None

    try: 
        conformation2_af_disordered_domains_idx, _ = get_af_disordered_domains(conformation2_af_initial_pred_path)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % conformation2_af_initial_pred_path) 
        print(e)
        print('SKIPING THIS INPUT...')
        return None, None

    conformation1_pdb_disordered_residues_idx, _ = get_pdb_disordered_residues(conformation1_af_seq, conformation1_pdb_seq, conformation1_af_disordered_domains_idx) 
    conformation2_pdb_disordered_residues_idx, _ = get_pdb_disordered_residues(conformation2_af_seq, conformation2_pdb_seq, conformation2_af_disordered_domains_idx) 
     
    print('CONFORMATION 1 PDB seq:')       
    print(conformation1_pdb_seq)
    print('CONFORMATION 2 PDB seq:')       
    print(conformation2_pdb_seq)

    conformation1_pdb_common_residues_idx, conformation2_pdb_common_residues_idx = get_residues_idx_in_seq1_and_seq2(conformation1_pdb_seq, conformation2_pdb_seq)
    conformation1_pdb_rmsf_below_threshold_idx, conformation2_pdb_rmsf_below_threshold_idx = get_residue_idx_below_rmsf_threshold(
                                                                                                conformation1_pdb_seq,
                                                                                                conformation2_pdb_seq,
                                                                                                conformation1_pdb_path,
                                                                                                conformation2_pdb_path,
                                                                                                'pdb',
                                                                                                'pdb',
                                                                                                conformation1_pdb_common_residues_idx,
                                                                                                conformation2_pdb_common_residues_idx,
                                                                                                conformation1_pdb_disordered_residues_idx,
                                                                                                conformation2_pdb_disordered_residues_idx)

    conformation1_pdb_common_residues_idx_complement = sorted(list(set(range(len(conformation1_pdb_seq))) - set(conformation1_pdb_common_residues_idx)))
    conformation2_pdb_common_residues_idx_complement = sorted(list(set(range(len(conformation2_pdb_seq))) - set(conformation2_pdb_common_residues_idx)))

    print('CONFORMATION 1 PDB common residues idx:')
    print(conformation1_pdb_common_residues_idx)
    print('CONFORMATION 1 PDB disordered:')
    print(conformation1_pdb_disordered_residues_idx)
    print('CONFORMATION 1 PDB intersection complement:')
    print(conformation1_pdb_common_residues_idx_complement)
    print('CONFORMATION 1 PDB below RMSF:')
    print(conformation1_pdb_rmsf_below_threshold_idx)

    print('CONFORMATION 2 PDB common residues idx:')
    print(conformation2_pdb_common_residues_idx)
    print('CONFORMATION 2 PDB disordered:')
    print(conformation2_pdb_disordered_residues_idx)
    print('CONFORMATION 2 PDB intersection complement:')
    print(conformation2_pdb_common_residues_idx_complement)
    print('CONFORMATION 2 PDB below RMSF:')
    print(conformation2_pdb_rmsf_below_threshold_idx)
    
    conformation1_pdb_residues_ignore_idx = conformation1_pdb_disordered_residues_idx+conformation1_pdb_common_residues_idx_complement+conformation1_pdb_rmsf_below_threshold_idx
    conformation2_pdb_residues_ignore_idx = conformation2_pdb_disordered_residues_idx+conformation2_pdb_common_residues_idx_complement+conformation2_pdb_rmsf_below_threshold_idx

    conformation1_pdb_residues_ignore_idx = tuple(sorted(set(conformation1_pdb_residues_ignore_idx)))
    conformation2_pdb_residues_ignore_idx = tuple(sorted(set(conformation2_pdb_residues_ignore_idx)))


    return conformation1_pdb_residues_ignore_idx, conformation2_pdb_residues_ignore_idx


def get_residues_ignore_idx_between_pdb_af_conformation(pdb_path, af_path, af_initial_pred_path):

    """Ignore residues based on:
        1. disorder
        2. common residues
        3. rmsf threshold

        af_initial_pred_path is to determine disordered residues
        and corresponds to a standard, vanila AF prediction
        while af_path corresponds to a perturbed prediction  

    """  

    pdb_seq = get_pdb_path_seq(pdb_path, None)
    af_seq = get_pdb_path_seq(af_initial_pred_path, None)

    '''try: 
        af_disordered_domains_idx, _ = get_af_disordered_domains(af_initial_pred_path)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % af_initial_pred_path) 
        print(e)
        print('SKIPING THIS INPUT...')
        return None

    pdb_disordered_residues_idx, _ = get_pdb_disordered_residues(af_seq, pdb_seq, af_disordered_domains_idx) 
    af_disordered_residues_idx = get_af_disordered_residues(af_disordered_domains_idx)''' 
     
    print('PDB seq:')       
    print(pdb_seq)
    print('AF seq:')       
    print(af_seq)

    pdb_common_residues_idx, af_common_residues_idx = get_residues_idx_in_seq1_and_seq2(pdb_seq, af_seq)

    pdb_rmsf_below_threshold_idx, af_rmsf_below_threshold_idx = get_residue_idx_below_rmsf_threshold(
                                                                                                pdb_seq,
                                                                                                af_seq,
                                                                                                pdb_path,
                                                                                                af_path,
                                                                                                'pdb',
                                                                                                'af',
                                                                                                pdb_common_residues_idx,
                                                                                                af_common_residues_idx,
                                                                                                [],
                                                                                                [])

    af_common_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_common_residues_idx)))

    print('AF common residues idx:')
    print(af_common_residues_idx)
    print('AF disordered:')
    print(af_disordered_residues_idx)
    print('AF intersection complement:')
    print(af_common_residues_idx_complement)
    print('AF below RMSF:')
    print(af_rmsf_below_threshold_idx)
    
    af_residues_ignore_idx = af_common_residues_idx_complement+af_rmsf_below_threshold_idx
    af_residues_ignore_idx = tuple(sorted(set(af_residues_ignore_idx)))


    return af_residues_ignore_idx


def get_comparison_residues_idx_between_pdb_af_conformation(pdb_path, af_initial_pred_path):

    """Compare residues based on:
        1. disorder
        2. common residues

        af_initial_pred_path is to determine disordered residues
        and corresponds to a standard, vanila AF prediction
        while af_path corresponds to a perturbed prediction  

    """  

    pdb_seq = get_pdb_path_seq(pdb_path, None)
    af_seq = get_pdb_path_seq(af_initial_pred_path, None)

    try: 
        af_disordered_domains_idx, _ = get_af_disordered_domains(af_initial_pred_path)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % af_initial_pred_path) 
        print(e)
        print('SKIPING THIS INPUT...')
        return None

    pdb_disordered_residues_idx, _ = get_pdb_disordered_residues(af_seq, pdb_seq, af_disordered_domains_idx) 
    af_disordered_residues_idx = get_af_disordered_residues(af_disordered_domains_idx)
     
    print('PDB seq:')       
    print(pdb_seq)
    print('AF seq:')       
    print(af_seq)

    pdb_common_residues_idx, af_common_residues_idx = get_residues_idx_in_seq1_and_seq2(pdb_seq, af_seq)
    af_common_residue_idx_ignoring_disordered_residues = [idx for idx in af_common_residues_idx if idx not in af_disordered_residues_idx]
    pdb_common_residue_idx_ignoring_disordered_residues = [idx for idx in pdb_common_residues_idx if idx not in pdb_disordered_residues_idx]
    
    print('AF common residues idx:')
    print(af_common_residues_idx)
    print('AF disordered:')
    print(af_disordered_residues_idx)
    print('PDB common residues idx:')
    print(pdb_common_residues_idx)
    print('PDB disordered:')
    print(pdb_disordered_residues_idx)


    return pdb_common_residue_idx_ignoring_disordered_residues, af_common_residue_idx_ignoring_disordered_residues


def get_contiguous_nonoverlapping_af_residues_idx_between_pdb_af_conformation(pdb_path, af_initial_pred_path):

    pdb_seq = get_pdb_path_seq(pdb_path, None)
    af_seq = get_pdb_path_seq(af_initial_pred_path, None)
    pdb_common_residues_idx, af_common_residues_idx = get_residues_idx_in_seq1_and_seq2(pdb_seq, af_seq)
    af_common_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_common_residues_idx)))
    
    af_common_residues_idx_complement_set = set(af_common_residues_idx_complement)
    af_contiguous_nonoverlapping_residues_idx = [idx for idx in af_common_residues_idx_complement if (idx+1) in af_common_residues_idx_complement_set or (idx-1) in af_common_residues_idx_complement_set]
     

    return af_contiguous_nonoverlapping_residues_idx


def get_residues_ignore_idx_between_af_conformations(conformation1_af_path, conformation2_af_path, af_initial_pred_path, rmsf_threshold=1.0):

    """Ignore residues based on:
        1. disorder
        2. rmsf threshold

        conformation1/2_af_path correspond to perturbed structures         

        Assumption: same sequence for all structures
        we use stricter rmsf threshold here 

    """  

    try: 
        af_disordered_domains_idx, _ = get_af_disordered_domains(af_initial_pred_path)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION') 
        print(e)
        af_disordered_domains_idx = [] 

    af_disordered_residues_idx = get_af_disordered_residues(af_disordered_domains_idx)

    conformation1_af_ca_pos = get_ca_coords_matrix(conformation1_af_path)
    conformation2_af_ca_pos = get_ca_coords_matrix(conformation2_af_path)

    ca_rmsf = np.linalg.norm(conformation2_af_ca_pos-conformation1_af_ca_pos, axis=1)
    af_rmsf_below_threshold_idx = list(np.where(ca_rmsf < rmsf_threshold)[0])

    print('Disordered residues index:')
    print(af_disordered_residues_idx)
    print('Resiudes below RMSF threshold index:')
    print(af_rmsf_below_threshold_idx)
    
    af_residues_ignore_idx = af_disordered_residues_idx+af_rmsf_below_threshold_idx
    af_residues_ignore_idx = tuple(sorted(set(af_residues_ignore_idx)))


    return af_residues_ignore_idx


def get_rmsf_pdb_af_conformation(pdb_path, af_path):

    print(pdb_path)
    print(af_path)

    pdb_seq = get_pdb_path_seq(pdb_path, None)
    af_seq = get_pdb_path_seq(af_path, None)

    pdb_common_residues_idx, af_common_residues_idx = get_residues_idx_in_seq1_and_seq2(pdb_seq, af_seq)
    out = get_rmsf(pdb_seq, af_seq, pdb_path, af_path, 'pdb', 'af', pdb_common_residues_idx, af_common_residues_idx)
    out_df = pd.DataFrame(out, columns=['pdb_residue_num_remapped','af_residue_num_remapped','pdb_residue_num_orig','af_residue_num_orig','pdb_residue_name','af_residue_name','rmsf']) 

    return out_df 
