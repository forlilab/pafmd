import argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO
import glob
import pandas as pd
import os
import pickle
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
import collections
from collections import defaultdict
import sys
import shutil 
from scipy.spatial.distance import pdist
from pre_md_refine import refine_multiple_rounds 
import json 
from pymol import cmd

from custom_openfold_utils.pdb_utils import get_pdb_path_seq, get_af_disordered_domains, get_af_disordered_residues, get_af_nterm_disorder_tail_idx, get_af_cterm_disorder_tail_idx, get_af_disordered_tail_residues, get_ca_pairwise_dist, convert_pdb_to_mmcif, get_bfactor


asterisk_line = '******************************************************************************'

def remove_files(file_list):
    for f in file_list:
        print('removing old file: %s' % f)
        os.remove(f)


def create_clustering_dist_matrix(ca_pdist_all):

    #method derived from
    #https://www.biorxiv.org/content/10.1101/2023.07.13.545008v2.supplementary-material

    clustering_dist_matrix = np.zeros((ca_pdist_all.shape[0],ca_pdist_all.shape[0]))
    for i in range(0,ca_pdist_all.shape[0]):
        for j in range(i+1,ca_pdist_all.shape[0]):
            pairwise_dist_diff = np.abs(ca_pdist_all[j,:] - ca_pdist_all[i,:])
            pairwise_dist_diff[pairwise_dist_diff <= 1] = 0.
            pairwise_dist_diff = np.sum(pairwise_dist_diff)
            clustering_dist_matrix[i,j] = pairwise_dist_diff
    
    return clustering_dist_matrix
 

def gen_md_input(conformation_info_dir, initial_pred_path, num_clusters, num_md_structures, plddt_threshold, disordered_threshold, remove_disordered_tails, max_candidate_conformations, overwrite, af_disordered_tail_residues_idx_custom=None):

    if plddt_threshold is None and disordered_threshold is None:
        md_structures_dir = '%s/md_starting_structures/num_clusters=%d/plddt_threshold=None/disordered_threshold=None' % (conformation_info_dir, num_clusters)
    elif plddt_threshold is not None and disordered_threshold is None:
        md_structures_dir = '%s/md_starting_structures/num_clusters=%d/plddt_threshold=%s/disordered_threshold=None' % (conformation_info_dir, num_clusters, str(plddt_threshold))
    elif plddt_threshold is None and disordered_threshold is not None:
        md_structures_dir = '%s/md_starting_structures/num_clusters=%d/plddt_threshold=None/disordered_threshold=%s' % (conformation_info_dir, num_clusters, str(disordered_threshold))
    else:
        md_structures_dir = '%s/md_starting_structures/num_clusters=%d/plddt_threshold=%s/disordered_threshold=%s' % (conformation_info_dir, num_clusters, str(plddt_threshold), str(disordered_threshold))

    print("CREATING DIR: %s" % md_structures_dir)
    os.makedirs(md_structures_dir, exist_ok=True)

    if af_disordered_tail_residues_idx_custom is None:
        try: 
            af_seq = get_pdb_path_seq(initial_pred_path, None)
            af_disordered_domains_idx, _ = get_af_disordered_domains(initial_pred_path)     
            nterm_disorder_start_idx, nterm_disorder_end_idx = get_af_nterm_disorder_tail_idx(af_disordered_domains_idx)
            cterm_disorder_start_idx, cterm_disorder_end_idx = get_af_cterm_disorder_tail_idx(af_disordered_domains_idx, len(af_seq))
            print('nterminal disordered tail idx: %s-%s' % (str(nterm_disorder_start_idx),str(nterm_disorder_end_idx)))
            print('cterminal disordered tail idx: %s-%s' % (str(cterm_disorder_start_idx),str(cterm_disorder_end_idx)))
            af_disordered_tail_residues_idx = get_af_disordered_tail_residues(af_disordered_domains_idx, len(af_seq))
            print('disordered tail residues idx:')
            print(af_disordered_tail_residues_idx)
            af_disordered_tail_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_disordered_tail_residues_idx)))
        except ValueError as e:
            print('TROUBLE PARSING AF PREDICTION %s' % initial_pred_path) 
            af_disordered_tail_residues_idx_complement = None
    else:
        af_seq = get_pdb_path_seq(initial_pred_path, None)
        af_disordered_tail_residues_idx = af_disordered_tail_residues_idx_custom
        af_disordered_tail_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_disordered_tail_residues_idx_custom))) 

    md_starting_structure_info_fname = '%s/md_starting_structure_info.pkl' % md_structures_dir
    pdb_files = glob.glob('%s/openmm_refined_structures/*.pdb' % md_structures_dir)
    json_files = glob.glob('%s/openmm_refined_structures/*.json' % md_structures_dir) 
    pdb_files.extend(glob.glob('%s/unrefined_structures/*.pdb' % md_structures_dir))

    if os.path.exists(md_starting_structure_info_fname):
        if not(overwrite):
            print("SKIPPING CLUSTERING PROCEDURE, %s ALREADY EXISTS:" % md_starting_structure_info_fname)
            return 
        else:
            remove_files(pdb_files)
            remove_files(json_files)
    else:
        remove_files(pdb_files)
        remove_files(json_files)

    pattern = "%s/conformation_info.pkl" % conformation_info_dir
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        pattern = "%s/**/conformation_info.pkl" % conformation_info_dir
        files = sorted(glob.glob(pattern, recursive=True))
        if len(files) == 0:
            print('conformation_info.pkl file(s) do not exist in dir: %s' % conformation_info_dir)
            sys.exit()
        else:
            print('%d conformation_info.pkl files found under dir: %s' % ((len(files), conformation_info_dir)))
    else:
        print('conformation_info file(s) found:')
        print(files)

    #combine into single dictionary
    for i in range(0,len(files)):
        with open(files[i], 'rb') as f:
            curr_conformation_info = pickle.load(f)
        if i == 0:
            conformation_info = curr_conformation_info.copy()
        else:
            curr_key = list(curr_conformation_info.keys())[0]
            if curr_key not in conformation_info:
                conformation_info.update(curr_conformation_info)
            else:
                conformation_info[curr_key].extend(curr_conformation_info[curr_key])

    add_conformations = True 
    conformation_info_all = [] 
    ca_pdist_all = []

    for key in conformation_info:
        for i,val in enumerate(conformation_info[key]):
            pdb_path = val[0]
            rmsd = val[1]
            mean_plddt = val[2]           
            disordered_percentage = val[3]            
            if max_candidate_conformations is not None:
                if len(conformation_info_all) >= max_candidate_conformations:
                    add_conformations = False
            if add_conformations:
                if plddt_threshold is None and disordered_threshold is None:
                    conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                    ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_tail_residues_idx_complement)
                    ca_pdist_all.append(ca_pdist)
                else:
                    if plddt_threshold is not None and disordered_threshold is None:
                        if mean_plddt >= plddt_threshold:
                            conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                            ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_tail_residues_idx_complement)
                            ca_pdist_all.append(ca_pdist)
                    elif plddt_threshold is None and disordered_threshold is not None:
                        if disordered_percentage <= disordered_threshold:
                            conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                            ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_tail_residues_idx_complement)
                            ca_pdist_all.append(ca_pdist)
                    elif plddt_threshold is not None and disordered_threshold is not None:
                        if mean_plddt >= plddt_threshold and disordered_percentage <= disordered_threshold:
                            conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                            ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_tail_residues_idx_complement)
                            ca_pdist_all.append(ca_pdist)



    total_candidate_conformations = len(conformation_info_all)
    print("%d TOTAL CANDIDATE CONFORMATIONS" % total_candidate_conformations)

    print("Building C-alpha pairwise distance matrix")     
    ca_pdist_all = np.array(ca_pdist_all) 
    clustering_dist_matrix = create_clustering_dist_matrix(ca_pdist_all)

    print("Performing Agglomerative Clustering") 
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage="average").fit(clustering_dist_matrix)

    print(asterisk_line)
    print('CLUSTER FREQUENCIES:')
    counter = collections.Counter(clustering.labels_)
    print(counter.most_common())
    print(asterisk_line)

    clustering_info_dict = {} 
    for idx,cluster_num in enumerate(clustering.labels_):
        if cluster_num in clustering_info_dict:
            clustering_info_dict[cluster_num].append(conformation_info_all[idx])
        else:
            clustering_info_dict[cluster_num] = [conformation_info_all[idx]] 

    for cluster_num in clustering_info_dict:
        clustering_info_dict[cluster_num] = sorted(clustering_info_dict[cluster_num], key=lambda x: x[2], reverse=True) #sort by plddt 

    cluster_idx_dict = {}
    for cluster_num,freq in counter.most_common():
        cluster_idx_dict[cluster_num] = 0 

    md_starting_structure_info_dict = defaultdict(list)
    curr_md_structures_count = 0
    
    while curr_md_structures_count < min(total_candidate_conformations,num_md_structures): 
        for cluster_num,freq in counter.most_common():
            curr_cluster_idx = cluster_idx_dict[cluster_num] 
            if len(clustering_info_dict[cluster_num]) > curr_cluster_idx and curr_md_structures_count < min(total_candidate_conformations,num_md_structures):
                md_starting_structure_info_dict[cluster_num].append(clustering_info_dict[cluster_num][curr_cluster_idx])
                cluster_idx_dict[cluster_num] += 1 
                curr_md_structures_count += 1
                print("Adding structure from cluster %d, index %d" % (cluster_num, cluster_idx_dict[cluster_num]-1)) 

    #add initial pred to md_starting_structure_info_dict
    cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
    _, mean_plddt = get_bfactor(cif_path)
    mean_plddt = round(mean_plddt,2)
    os.remove(cif_path)
    md_starting_structure_info_dict['initial'] = [[initial_pred_path, 0, mean_plddt]]
     
    print(asterisk_line)
    print('MD STARTING STRUCTURE INFO:')
    for key, value in md_starting_structure_info_dict.items():
        print(f"Cluster number: {key}, Number of elements: {len(value)}")
    print(asterisk_line)

    structural_issues_info_summary = {} 

    md_starting_structure_info_dict_keys = list(md_starting_structure_info_dict.keys())
    md_starting_structure_info_dict_keys.remove('initial')
    md_starting_structure_info_dict_keys.insert(0,'initial')

    unrefined_output_dir = '%s/unrefined_structures' % md_structures_dir
    refined_output_dir = '%s/openmm_refined_structures' % md_structures_dir
    os.makedirs(unrefined_output_dir, exist_ok=True)
    os.makedirs(refined_output_dir, exist_ok=True)         


    for cluster_num in md_starting_structure_info_dict_keys:

        for i in range(0,len(md_starting_structure_info_dict[cluster_num])): 

            if cluster_num != 'initial':
                pdb_source_path = md_starting_structure_info_dict[cluster_num][i][0]
                plddt = md_starting_structure_info_dict[cluster_num][i][2]
                output_fname = 'cluster_%d_idx_%d_plddt_%s' % (cluster_num, i, str(round(plddt)))
            else:
                pdb_source_path = md_starting_structure_info_dict[cluster_num][i][0]
                output_fname = 'initial_pred' 
 
            pdb_target_path = '%s/%s.pdb' % (unrefined_output_dir, output_fname)
            shutil.copyfile(pdb_source_path, pdb_target_path)
    
            if remove_disordered_tails:
                cmd.reinitialize()
                cmd.load(pdb_target_path, 'protein')
                if len(af_disordered_tail_residues_idx) > 0:
                    disordered_tail_residues_str = [str(res+1) for res in af_disordered_tail_residues_idx]
                    residue_selection = '+'.join(disordered_tail_residues_str)
                    cmd.remove(f"resi {residue_selection}")                
                    print(f"removed residues: {disordered_tail_residues_str}")
                    cmd.save(pdb_target_path, 'protein')                      
                cmd.delete('all')
               
            md_unrefined_structure_path = os.path.abspath(pdb_target_path)
            md_refined_structure_path = md_unrefined_structure_path.replace('.pdb', '_openmm_refinement.pdb')
            md_refined_structure_path = md_refined_structure_path.replace('unrefined_structures','openmm_refined_structures')
            print('****************************************')
            print('****************************************')
            print("ATTEMPTING TO RESOLVE ANY STRUCTURAL ISSUES WITH %s" % md_unrefined_structure_path)
            print('****************************************')
            print('****************************************')
            
            if cluster_num == 'initial':
                structural_issues_info = refine_multiple_rounds(md_unrefined_structure_path, md_refined_structure_path, add_disulfide_bonds=True, disulfide_bond_residues_idx=None)
                disulfide_bond_residues_idx_derived_from_initial_pred = structural_issues_info['disulfide_bond_ids']
            else:
                if len(disulfide_bond_residues_idx_derived_from_initial_pred) > 0:
                    structural_issues_info = refine_multiple_rounds(md_unrefined_structure_path, md_refined_structure_path, add_disulfide_bonds=True, disulfide_bond_residues_idx=disulfide_bond_residues_idx_derived_from_initial_pred)
                else:
                    structural_issues_info = refine_multiple_rounds(md_unrefined_structure_path, md_refined_structure_path, add_disulfide_bonds=False)

            md_starting_structure_info_dict[cluster_num][i].append(md_refined_structure_path)
            structural_issues_info_summary[md_refined_structure_path] = structural_issues_info

    print(structural_issues_info_summary)

    output_file = '%s/structural_issues_info_summary.json' % md_structures_dir
    print('saving %s' % output_file)
    with open(output_file, 'w') as json_file:
        json.dump(structural_issues_info_summary, json_file, sort_keys=True, indent=4)
    
    md_starting_structure_info_fname = '%s/md_starting_structure_info.pkl' % md_structures_dir
    with open(md_starting_structure_info_fname, 'wb') as f:
        pickle.dump(md_starting_structure_info_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conformation_info_dir", type=str, default=None,
        help="Either points directly to the parent directory containing conformation_info.pkl or an nth-parent directory if multiple conformation_info.pkl are present",
    )
    parser.add_argument(
        "--initial_pred_path", type=str, default=None,
        help="Points directly to the vanila AF prediction",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10,
        help="",
    )
    parser.add_argument(
        "--num_md_structures", type=int, default=50,
        help="number of structures to use as starting points for MD"
    )
    parser.add_argument(
        "--plddt_threshold", type=float, default=70,
        help="only clusters conformations whose plddt score is greater than this threshold (which is a value between 0-100)",
    )
    parser.add_argument(
        "--disordered_threshold", type=float, default=70,
        help="only clusters conformations whose disordered percentage is less than this threshold (which is a value between 0-100)",
    )
    parser.add_argument(
        "--remove_disordered_tails", action="store_true", default=False
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False
    )
    parser.add_argument(
        "--max_candidate_conformations", type=int, default=None, 
        help="maximum number of conformations to use as candidate pool for selecting MD starting structures"
    )
    parser.add_argument('--af_disordered_tail_residues_idx_custom', nargs='+', default=None,
        help='List of prespecified disordered tail residues indices to remove')


    args = parser.parse_args()

    gen_md_input(args.conformation_info_dir, args.initial_pred_path, args.num_clusters, args.num_md_structures, args.plddt_threshold, args.disordered_threshold, args.remove_disordered_tails, args.max_candidate_conformations, args.overwrite, args.af_disordered_tail_residues_idx_custom)
