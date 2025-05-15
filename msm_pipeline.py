import sys 
import glob
import os
import subprocess
import re  
import shutil
import pickle
import argparse
from typing import Tuple, List, Mapping, Optional, Sequence, Any, MutableMapping, Union
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import self_distance_array
from MDAnalysis.analysis import rms

from custom_openfold_utils.pdb_utils import superimpose_wrapper_monomer, get_rmsd, tmalign_wrapper


asterisk_line = '******************************************************************************'

gen_msm_summary_path = os.path.abspath('./gen_msm_summary.py')
path_to_msm_clustering_exe = os.path.abspath('./Clustering/build/clustering')

def list_of_strings(arg):
    return arg.split(',')

def dump_pkl(data, fname, output_dir):
    output_path = '%s/%s.pkl' % (output_dir, fname)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def remove_files_in_dir(path):
    file_list = sorted(glob.glob('%s/*' % path))
    for f in file_list:
        if not os.path.isdir(f):
            print('removing old file: %s' % f)
            os.remove(f)

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Output:")
    print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:")
        print(result.stderr)


def get_msm_input_data(
    input_prmtop_path: str, 
    input_traj_path: str, 
    output_dir: str, 
    max_frames: int = None, 
    initial_pred_path: int = None, 
    write_reimaged_traj: bool =False
):

    """Reimages protein and calculates alpha-carbon pairwise distances and contacts.  
    
       If initial pred_path is provided, the trajectory is aligned to it, 
       each aligned MD frame is saved and RMSD/RMSF metrics are calculated. 

       Reimaged trajectory is written if write_reimaged_traj is set to True.   
    """

    os.makedirs(output_dir, exist_ok=True)

    print('loading %s' % input_traj_path)
    whole_system_u = mda.Universe(input_prmtop_path, input_traj_path, select='protein and not resname HOH')

    num_frames_per_traj = len(whole_system_u.trajectory)
    timestep_ns = whole_system_u.trajectory.dt/1000
    print('total frames: %d' % num_frames_per_traj)
    print('timestep (ns): %f' % timestep_ns)

    if max_frames is not None:
        if max_frames <= num_frames_per_traj:
            num_frames_per_traj = max_frames
        else:
            print('ERROR: max frames greater than num frames')
            sys.exit()

    protein = whole_system_u.select_atoms('protein')

    protein_coords = np.empty((num_frames_per_traj, protein.atoms.n_atoms, 3))
    boxes = np.empty((num_frames_per_traj, 6))
    for ts in whole_system_u.trajectory[0:num_frames_per_traj]:
        boxes[ts.frame] = ts.dimensions
        protein_coords[ts.frame] = protein.atoms.positions
    boxes = boxes[0]

    print('protein coordinates shape')
    print(protein_coords.shape)

    protein_u = mda.Merge(protein.atoms).load_new(protein_coords, dimensions=boxes)
    protein_u.add_TopologyAttr('tempfactors')

    #these transformations are lazy
    transforms = [
        trans.unwrap(protein_u.atoms),
        trans.center_in_box(protein_u.atoms, center='mass'),
        trans.wrap(protein_u.atoms, compound='fragments'),
    ]

    print('reimaging protein')
    protein_u.trajectory.add_transformations(*transforms)

    protein_u.add_TopologyAttr("chainID")
    for atom in protein_u.atoms:
        atom.chainID = 'A'

    rmsd_df = [] 

    if initial_pred_path is not None:
        print('aligning to %s' % initial_pred_path)
        ref = mda.Universe(initial_pred_path)
        ref_protein = ref.select_atoms("protein")
        aligner = align.AlignTraj(protein_u, ref_protein, select='name CA', in_memory=True)
        aligner.run() 

        #rmsd and rmsf calculated w.r.t initial_pred_path 
        print('calculating rsmd')
        rmsd_calc = rms.RMSD(protein_u, ref_protein, select='name CA', ref_frame=0)
        rmsd_calc.run()
        rmsd_results = rmsd_calc.results.rmsd 
 
        print('calculating rmsf')
        mobile = protein_u.select_atoms('name CA')
        rmsf_calc = rms.RMSF(mobile).run()
        rmsf_results = rmsf_calc.results.rmsf
        rmsf_dict = dict(zip(mobile.resids, rmsf_results))

        cluster_num = output_dir[output_dir.rindex('/')+1:]

        aligned_frames_save_dir = '%s/aligned_frames_wrt_initial_pred' % output_dir
        os.makedirs(aligned_frames_save_dir, exist_ok=True)
        remove_files_in_dir(aligned_frames_save_dir)
        
        for ts in protein_u.trajectory: 
            frame_num = ts.frame 
            rmsd = rmsd_results[frame_num][2]
            rmsd_str = str(round(rmsd,2)).replace('.','-')        
            output_pdb_path = '%s/frame%d-rmsd_wrt_initial-%s.pdb' % (aligned_frames_save_dir, frame_num, rmsd_str)
            protein_u.atoms.tempfactors = 0.0
            for residue in protein_u.residues:
                ca_atom = residue.atoms.select_atoms('name CA')
                if len(ca_atom) > 0:
                    residue.atoms.tempfactors = rmsf_dict[residue.resid] 
            if frame_num % 10 == 0:
                print('saving %s' % output_pdb_path)
            protein_u.atoms.write(output_pdb_path)
            rmsd_df.append([frame_num, cluster_num, initial_pred_path, output_pdb_path, round(rmsd,3)])
 
    if write_reimaged_traj:
        print('writing reimaged trajectory')
        output_traj_path = '%s/trajectory-centered.nc' % output_dir
        with mda.Writer(output_traj_path, protein_u.atoms.n_atoms) as W:
            for ts in protein_u.trajectory:
                if ts.frame == 0:
                    box_dimensions = ts.dimensions
                    print(f"Box dimensions (a, b, c, alpha, beta, gamma) = {box_dimensions}")
                if ts.frame % 100 == 0:
                    print(f"Writing frame {ts.frame}/{num_frames_per_traj}")
                W.write(protein_u.atoms)

    print('calculating ca pairwise distances and contacts')
    ca_atoms = protein_u.select_atoms('name CA')
    num_atoms = len(ca_atoms)

    ca_pdist = np.zeros((num_frames_per_traj, num_atoms * (num_atoms - 1) // 2))

    for ts in protein_u.trajectory:
        pdist = self_distance_array(ca_atoms.positions) 
        ca_pdist[ts.frame] = pdist

    print('pairwise distance matrix shape:')
    print(ca_pdist.shape)

    ca_contacts = (ca_pdist <= 8.0).astype(int)

    #https://www.nature.com/articles/s41467-024-53170-z#Sec9
    #####contacts with a peak-to-peak value (range) versus mean ratio of less than 0.2 are considered static
    dist_ratio = np.ptp(ca_pdist, axis=0) / (np.mean(ca_pdist, axis=0) + 1e-10)
    static_contacts_idx = np.where(dist_ratio < .2)[0] 

    print('%d static contacts found out of %d total contacts' % (len(static_contacts_idx), ca_contacts.shape[1]))
    
    protein_u.trajectory[0]
    output_pdb_path = '%s/protein-reimaged.pdb' % output_dir
    print('saving %s' % output_pdb_path)
    protein_u.atoms.write(output_pdb_path)


    return ca_pdist, ca_contacts, static_contacts_idx, rmsd_df, num_frames_per_traj, timestep_ns 


def run_pca(
    input_data: np.ndarray,
    output_dir: str,
    output_fname: str
):
    """Runs PCA on input data. Saves first 3 PC components as csv file"""

    print('Running PCA')
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    pca = PCA(n_components=10)
    pc_coords = pca.fit_transform(input_data_scaled)
    
    pca_df = pd.DataFrame({'pc1': pc_coords[:,0], 'pc2': pc_coords[:,1], 'pc3': pc_coords[:,2]})
    explained_variance_df = pd.DataFrame({
       'component_num': [f'PC{i+1}' for i in range(pca.n_components_)],
       'explained_var': pca.explained_variance_ratio_
    })

    print('explained variance:')
    print(explained_variance_df)

    explained_var_path = '%s/%s_explained_var.csv' % (output_dir, output_fname)
    print('saving %s' % explained_var_path)
    explained_variance_df.to_csv(explained_var_path, index=False)

    pca_output_path = '%s/%s_numpc=3' % (output_dir, output_fname)
    print('saving %s' % pca_output_path)
    pca_df.to_csv(pca_output_path, sep=' ', index=False, header=False)

    return pca_output_path 


def run_msm_commands(
    pca_data_path: str, 
    rmsd_info_path: str, 
    exp_pdb_dict_path: str, 
    num_frames_per_traj: int, 
    num_trajectories: int, 
    timestep_ns: float, 
    output_dir: str
):
    """Runs markov state model (MSM) pipeline that maps MD frames to corresponding microstates and macrostates. 
       See https://moldyn.github.io/Clustering/docTutorial.html for more technical details of this pipeline. 
       The specific pipeline we are using can be found here: https://github.com/moldyn/HP35/blob/main/CLUSTERING/perform_clustering (doi: 10.1021/acs.jctc.3c00240)
    """ 

    total_frames = num_trajectories*num_frames_per_traj
    sim_time_ns = num_frames_per_traj*timestep_ns
    print('each trajectory run for %.2f ns' % sim_time_ns)
    if sim_time_ns >= 100:
        print('using lagtime of 10 ns')
        lag_steps = int(10/timestep_ns) #corresponds to lagtime of 10ns 
    else:
        print('using lagtime of 1 ns')
        lag_steps = int(1/timestep_ns) 

    min_population = int(total_frames/100) #corresponds to the minimum number of frames per microstate  

    print('changing directory to %s' % output_dir)
    os.chdir(output_dir) 
   
    #this calculates a free energy for each conformation with a density based clustering method  
    free_energy_output_path = '%s/free_energy' % output_dir
    nearest_neighbor_output_path = '%s/nearest_neighbor' % output_dir 
    command = [
        path_to_msm_clustering_exe,
        "density",             
        "-f", pca_data_path,
        "-pop",               
        "-d", free_energy_output_path,
        "-b", nearest_neighbor_output_path,
        "-v"              
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    #assigns each frame to a cluster which corresponds to geometrically connected regions with values below a free energy cutoff 
    cluster_info_output_dir = '%s/cluster_info' % output_dir
    os.makedirs(cluster_info_output_dir, exist_ok=True)
    command = [
        path_to_msm_clustering_exe,
        "density",             
        "-f", pca_data_path,
        "-T", "-1",
        "-D", free_energy_output_path,
        "-B", nearest_neighbor_output_path,
        "-o", '%s/cluster' % cluster_info_output_dir,
        "-v"
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    print('changing directory to: %s' % cluster_info_output_dir)
    os.chdir(cluster_info_output_dir)

    #identifies local minima of free energy landscape based on if a cluster is geometrically disconnected, 
    #has a population above min_population and is not at the highest free energy  
    command = [
        path_to_msm_clustering_exe,
        "network",
        "-p", str(min_population), 
        "-b", "cluster",
        "-v"
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    os.chdir(output_dir)

    #assigns each frame to a microstate by iteratively assigning frames with the current 
    #lowest free energy to the nearest free energy minima basin 
    microstate_output_dir = '%s/microstate_info' % output_dir 
    os.makedirs(microstate_output_dir, exist_ok=True)
    microstate_output_path = '%s/microstates_pc3_minpopulation=%d' % (microstate_output_dir, min_population)
    command = [
        path_to_msm_clustering_exe,
        "density",             
        "-f", pca_data_path,
        "-i", '%s/network_end_node_traj.dat' % cluster_info_output_dir,
        "-D", free_energy_output_path,
        "-B", nearest_neighbor_output_path,
        "-o", microstate_output_path,
        "-v" 
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    
    #clusters microstates into macrostates based on their respective transition probabilities
    #transition matrix is calculated according to specified lag steps 
    #if self-transition probability is lower than certain metastability criterion
    #state will be lumped with state to which transition probability is highest 
    mpp_output_path = '%s/microstates_pc3_minpopulation=%d_lagsteps=%d' % (microstate_output_dir, min_population, lag_steps) 
    command = [
        path_to_msm_clustering_exe,
        "mpp",             
        "-s", microstate_output_path,
        "-l", str(lag_steps), 
        "-D", free_energy_output_path,
        "-o", mpp_output_path,
        "--concat-nframes", str(num_frames_per_traj), 
        "--concat-limits", str(num_trajectories),
        "-v"
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    #pipeline to refine macrostate assignments 
    #and ouptut relevant plots (dendrograms, pymol sessions, and markov state model QC)
    linkage_matrix_path = '%s_transitions.dat' % mpp_output_path
    macrostate_info_output_dir = '%s/macrostate_info' % output_dir 

    arg1 = '--linkage_matrix_path=%s' % linkage_matrix_path
    arg2 = '--microstate_info_path=%s' % microstate_output_path
    arg3 = '--rmsd_info_path=%s' % rmsd_info_path
    arg4 = '--output_dir=%s' % macrostate_info_output_dir
    arg5 = '--lag_steps=%d' % lag_steps
    arg6 = '--timestep_ns=%f' % timestep_ns
    arg7 = '--num_frames_per_traj=%d' % num_frames_per_traj
    arg8 = '--save_publication_session'
    script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8]

    if exp_pdb_dict_path is not None:
        arg9 = '--exp_pdb_dict_path=%s' % exp_pdb_dict_path
        script_arguments.append(arg9)

    cmd_to_run = ["python", gen_msm_summary_path] + script_arguments
    cmd_to_run_str = ' '.join(cmd_to_run)
    print('running command: %s' % cmd_to_run_str)
    subprocess.run(cmd_to_run)

def calc_rmsd_tmscore(
    msm_output_dir: str, 
    TMalign_path: str
):

    """Calculates RMSD and TM-score between MD frames and experimental PDB structures. 
       TM-score is only calculated if path to TM-align exe is provided 
    """ 

    rmsd_info_path = '%s/rmsd_df.csv' % msm_output_dir 
    rmsd_df = pd.read_csv(rmsd_info_path)

    aligned_exp_pdb_save_dir = '%s/aligned_exp_pdb_wrt_initial_pred' % msm_output_dir
    exp_pdb_dict_path = '%s/exp_pdb_dict.pkl' % aligned_exp_pdb_save_dir
    with open(exp_pdb_dict_path, 'rb') as file:
        exp_pdb_dict = pickle.load(file)

    for exp_pdb_id in exp_pdb_dict:
        exp_pdb_path = exp_pdb_dict[exp_pdb_id]
        rmsd_list = []
        tmscore_list = [] 
        for idx,md_frame_path in enumerate(list(rmsd_df['md_frame_path'])):
            frame_num = list(rmsd_df['frame_num'])[idx]
            if frame_num % 10 != 0:
                print('calculating rmsd/tm-score between %s and %s' % (md_frame_path, exp_pdb_path))
            rmsd = round(get_rmsd(exp_pdb_path, md_frame_path, pdb1_chain='A', pdb2_chain='A'),3)
            rmsd_list.append(rmsd)
            if TMalign_path is not None:
                tm_score = round(tmalign_wrapper(exp_pdb_path, md_frame_path, TMalign_path),3)
                tmscore_list.append(tm_score)

        colname = 'rmsd_wrt_%s' % exp_pdb_id
        rmsd_df[colname] = rmsd_list
        if len(tmscore_list) > 0:
            colname = 'tmscore_wrt_%s' % exp_pdb_id
            rmsd_df[colname] = tmscore_list
        print('saving %s' % rmsd_info_path)
        rmsd_df.to_csv(rmsd_info_path, index=False) 


def run_msm_pipeline_only(
    uniprot_id: str, 
    traj_paths: List[str], 
    initial_pred_path: str, 
    exp_pdb_ids: List[str], 
    save_dir: str, 
    TMalign_path: str,
    calc_similarity_md_frames_exp_pdb: bool = False,
    max_frames: int = None
):

    """Runs MSM pipeline that maps MD frames to corresponding microstates and macrostates. 
       Assumes relevant input files have already been generated. 
    """ 

    num_trajectories = len(traj_paths)
    msm_output_dir = '%s/msm_pipeline_output' % save_dir

    input_traj_path = traj_paths[0]
    input_prmtop_path = input_traj_path.replace('trajectory.nc', 'minimization_round2.prmtop')
    print('loading %s' % input_traj_path)
    whole_system_u = mda.Universe(input_prmtop_path, input_traj_path, select='protein and not resname HOH')
    num_frames_per_traj = len(whole_system_u.trajectory)
    timestep_ns = whole_system_u.trajectory.dt/1000
    print('total frames: %d' % num_frames_per_traj)
    print('timestep (ns): %f' % timestep_ns)

    #columns of this include: 
    ##### 'frame_num', 'cluster_num', 'initial_pred_path', 'md_frame_path', 'rmsd_wrt_initial_pred' 
    rmsd_info_path = '%s/rmsd_df.csv' % msm_output_dir 
    if not(os.path.exists(rmsd_info_path)):
        print('%s does not exist, exiting...' % rmsd_info_path)
        sys.exit()

    aligned_exp_pdb_save_dir = '%s/aligned_exp_pdb_wrt_initial_pred' % msm_output_dir
    if os.path.exists(aligned_exp_pdb_save_dir):
        shutil.rmtree(aligned_exp_pdb_save_dir)

    #if exp_pdb_ids are passed, the corresponding pdbs are fetched and aligned to initial_pred_path 
    exp_pdb_dict = {} #maps exp_pdb_id to pdb_path 
    exp_pdb_dict_path = None 
    for exp_pdb_id in exp_pdb_ids:
        aligned_exp_pdb_save_dir = '%s/aligned_exp_pdb_wrt_initial_pred' % msm_output_dir
        os.makedirs(aligned_exp_pdb_save_dir, exist_ok=True)
        _, _, _, exp_pdb_path_aligned = superimpose_wrapper_monomer(None, exp_pdb_id, 'pred', 'pdb', initial_pred_path, None, aligned_exp_pdb_save_dir, clean=True)
        exp_pdb_dict[exp_pdb_id] = exp_pdb_path_aligned 
        exp_pdb_dict_path = '%s/exp_pdb_dict.pkl' % aligned_exp_pdb_save_dir
        print('saving %s' % exp_pdb_dict_path)
        dump_pkl(exp_pdb_dict, 'exp_pdb_dict', aligned_exp_pdb_save_dir)

    if calc_similarity_md_frames_exp_pdb:
        calc_rmsd_tmscore(msm_output_dir, TMalign_path)
 
    pca_output_dir = '%s/pca_output' % save_dir        
    pca_pdist_path = '%s/%s_numpc=3' % (pca_output_dir, 'pca_pdist')
    pca_contacts_path = '%s/%s_numpc=3' % (pca_output_dir, 'pca_ca_contacts')

    if not(os.path.exists(pca_pdist_path)):
        print('%s does not exist, exiting...' % pca_pdist_path) 
    if not(os.path.exists(pca_contacts_path)):
        print('%s does not exist, exiting...' % pca_contacts_path) 

    msm_output_dir = '%s/msm_pipeline_output/pca_pdist' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
    run_msm_commands(pca_pdist_path, rmsd_info_path, exp_pdb_dict_path, num_frames_per_traj, num_trajectories, timestep_ns, msm_output_dir)

    msm_output_dir = '%s/msm_pipeline_output/pca_contacts' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
    run_msm_commands(pca_contacts_path, rmsd_info_path, exp_pdb_dict_path, num_frames_per_traj, num_trajectories, timestep_ns, msm_output_dir)



def run_full_pipeline(
    uniprot_id: str, 
    traj_paths: List[str], 
    initial_pred_path: str, 
    exp_pdb_ids: List[str], 
    save_dir: str,
    TMalign_path: str,
    calc_similarity_md_frames_exp_pdb: bool = False, 
    max_frames: int = None
):

    """Runs full pipeline that 
          1) assembles input coordinates (PCA CA pairwise distances/contacts)
          2) runs MSM pipeline that maps MD frames to corresponding microstates and macrostates

       If exp_pdb_ids are passed, the corresponding pdbs are fetched and aligned to initial_pred_path. 
       If calc_similarity_md_frames_exp_pdb is True, each MD frame is aligned to each exp_pdb_id and saved.  
    """ 

    num_trajectories = len(traj_paths)

    ca_pdist_all_traj = []
    ca_contacts_all_traj = []
    rmsd_df_all_traj = []
    static_contacts_idx_all_traj = []

    for i in range(0,len(traj_paths)):
        print('on path %s (%d/%d)' % (traj_paths[i], i+1, len(traj_paths)))
        input_traj_path = traj_paths[i]
        input_prmtop_path = input_traj_path.replace('trajectory.nc', 'minimization_round2.prmtop')
        output_dir = input_traj_path[0:input_traj_path.rindex('/')] 
        ca_pdist, ca_contacts, static_contacts_idx, rmsd_df, num_frames_per_traj, timestep_ns = get_msm_input_data(input_prmtop_path, input_traj_path, output_dir, initial_pred_path=initial_pred_path)
        prev_num_frames_per_traj = num_frames_per_traj
        static_contacts_idx_all_traj.append(static_contacts_idx)
        rmsd_df_all_traj.extend(rmsd_df) 
           
        if i == 0:
            ca_pdist_all_traj.append(ca_pdist)
            ca_contacts_all_traj.append(ca_contacts)
        else:
            if prev_num_frames_per_traj != num_frames_per_traj:
                print('WARNING: %s does not match number of frames (%d vs %d) from previous trajectory, so ignoring from analysis' % (input_traj_path, prev_num_frames_per_traj, num_frames_per_traj))
            else:
                ca_pdist_all_traj.append(ca_pdist)
                ca_contacts_all_traj.append(ca_contacts)
 
    msm_output_dir = '%s/msm_pipeline_output' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
 
    rmsd_df_all_traj = pd.DataFrame(rmsd_df_all_traj, columns = ['frame_num', 'cluster_num', 'initial_pred_path', 'md_frame_path', 'rmsd_wrt_initial_pred'])
    rmsd_df_all_traj.insert(0, 'uniprot_id', uniprot_id)
    print(rmsd_df_all_traj)
    rmsd_info_path = '%s/rmsd_df.csv' % msm_output_dir 
    print('saving %s' % rmsd_info_path)
    rmsd_df_all_traj.to_csv(rmsd_info_path, index=False) 
   
    #if exp_pdb_ids are passed, the corresponding pdbs are fetched and aligned to initial_pred_path 
    exp_pdb_dict = {} #maps exp_pdb_id to pdb_path 
    exp_pdb_dict_path = None 
    for exp_pdb_id in exp_pdb_ids:
        aligned_exp_pdb_save_dir = '%s/aligned_exp_pdb_wrt_initial_pred' % msm_output_dir
        os.makedirs(aligned_exp_pdb_save_dir, exist_ok=True)
        _, _, _, exp_pdb_path_aligned = superimpose_wrapper_monomer(None, exp_pdb_id, 'pred', 'pdb', initial_pred_path, None, aligned_exp_pdb_save_dir, clean=True)
        exp_pdb_dict[exp_pdb_id] = exp_pdb_path_aligned 
        exp_pdb_dict_path = '%s/exp_pdb_dict.pkl' % aligned_exp_pdb_save_dir
        print('saving %s' % exp_pdb_dict_path)
        dump_pkl(exp_pdb_dict, 'exp_pdb_dict', aligned_exp_pdb_save_dir)

    if calc_similarity_md_frames_exp_pdb:
        calc_rmsd_tmscore(msm_output_dir, TMalign_path)
 
    #static contacts that exist among all simulations 
    print('calculating common subset of static contacts among all trajectories')
    static_contacts_idx_intersection = reduce(np.intersect1d, static_contacts_idx_all_traj)

    ca_pdist_all_traj = np.array(ca_pdist_all_traj)
    ca_contacts_all_traj = np.array(ca_contacts_all_traj) 

    ca_pdist_all_traj = np.concatenate(ca_pdist_all_traj, axis=0)
    ca_contacts_all_traj = np.concatenate(ca_contacts_all_traj, axis=0)

    print(ca_contacts_all_traj.shape)
    print('removing %d static contacts out of %d total contacts' % (len(static_contacts_idx_intersection), ca_contacts_all_traj.shape[1])) 
    ca_dynamic_contacts_all_traj = np.delete(ca_contacts_all_traj, static_contacts_idx_intersection, axis=1) 
    print(ca_dynamic_contacts_all_traj.shape)
                 
    pca_output_dir = '%s/pca_output' % save_dir
    os.makedirs(pca_output_dir, exist_ok=True)

    pca_info_fname = '%s/pca_info.txt' % pca_output_dir
    with open(pca_info_fname, 'w') as f:
        f.write("shape of ca_pairwise_dist matrix: \n")
        f.write(str(ca_pdist_all_traj.shape))
        f.write("\n")
        f.write("shape of ca_contacts matrix: \n")
        f.write(str(ca_dynamic_contacts_all_traj.shape))
        f.write("\n")
         
    output_fname = 'pca_pdist'
    pca_pdist_path = run_pca(ca_pdist_all_traj, pca_output_dir, output_fname)               
    output_fname = 'pca_ca_contacts'
    pca_contacts_path = run_pca(ca_dynamic_contacts_all_traj, pca_output_dir, output_fname)

    msm_output_dir = '%s/msm_pipeline_output/pca_pdist' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
    run_msm_commands(pca_pdist_path, rmsd_info_path, exp_pdb_dict_path, num_frames_per_traj, num_trajectories, timestep_ns, msm_output_dir)

    msm_output_dir = '%s/msm_pipeline_output/pca_contacts' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
    run_msm_commands(pca_contacts_path, rmsd_info_path, exp_pdb_dict_path, num_frames_per_traj, num_trajectories, timestep_ns, msm_output_dir)


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uniprot_id", type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--traj_parent_dir", type=str, default=None,
        help="Parent directory where trajectory files of extensions .nc are saved. Trajectory files are retrieved across all subdirectories of traj_parent_dir",
    )
    parser.add_argument(
        "--initial_pred_path", type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--exp_pdb_ids", type=list_of_strings, default=None,
        help="If you want to retrieve experimentally determined reference pdbs to supplement your analysis, feed in a list of strings separated by commas where strings are of the format XXX_Y, where XXXX is the pdb_id and Y is the chain_id"
    )
    parser.add_argument(
        "--calc_similarity_md_frames_exp_pdb", action='store_true', default=False,
        help="If true, calculates RMSD and TM-score between MD frames and experimentally determined reference pdbs (as specified by exp_pdb_ids)"
    )
    parser.add_argument(
        "--TMalign_path", type=str, default=None,
        help="If calc_similarity_md_frames_exp_pdb set to True and TM-score is to be calculated, pass in path to TMalign exe"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="If you only want to use the first N frames per simulation for analysis, pass in an integer value to this argument",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--skip_input_coordinate_gen", action='store_true', default=False,
        help="If true, assumes input coordinates (PCA CA pairwise distances/contacts) have been generated already and only runs markov state model portion of pipeline"
    )


    args = parser.parse_args()

    if args.exp_pdb_ids is not None:
        for pdb in args.exp_pdb_ids:
            if len(pdb.split('_')) != 2:
                raise ValueError("Every entry in the pdb_id_list must be of format XXXX_Y, where XXXX is the pdb_id and Y is the chain_id")
    else:
        args.exp_pdb_ids = [] 

    traj_paths = sorted(glob.glob('%s/**/trajectory.nc' % args.traj_parent_dir))

    if len(traj_paths) == 0:
        raise ValueError("No trajectory.nc files found in %s" % args.traj_parent_dir) 

    if not(args.skip_input_coordinate_gen):
        run_full_pipeline(args.uniprot_id, traj_paths, args.initial_pred_path, args.exp_pdb_ids, args.save_dir, args.TMalign_path, args.calc_similarity_md_frames_exp_pdb, args.max_frames) 
    else:
        run_msm_pipeline_only(args.uniprot_id, traj_paths, args.initial_pred_path, args.exp_pdb_ids, args.save_dir, args.TMalign_path, args.calc_similarity_md_frames_exp_pdb, args.max_frames) 

