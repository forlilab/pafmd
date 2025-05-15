#!/usr/bin/env python3
#sourced from https://github.com/moldyn/HP35/blob/main/MPP/process_mpp.py
"""Generate Dendrogram from MPP and Lump."""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import os 
import shutil 
import argparse
import pickle
from itertools import chain 
from functools import lru_cache
from typing import Tuple, List, Mapping, Optional, Sequence, Any, MutableMapping, Union
import msmhelper as mh
import pandas as pd 
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, Normalize, LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import colorsys
sys.setrecursionlimit(20000)
np.set_printoptions(threshold=sys.maxsize)

from pymol import cmd

from custom_openfold_utils.conformation_utils import get_contiguous_nonoverlapping_af_residues_idx_between_pdb_af_conformation
 

#both vars are for dendrogram
MIN_EDGE_WEIGHT = .2 
EDGE_WEIGHT_SCALING_FACTOR=20 

asterisk_line = '******************************************************************************'


def get_b_factor_percentile(selection, percentile=95):
    b_factors = []
    cmd.iterate(selection, "b_factors.append(b)", space=locals())
    return np.percentile(b_factors, percentile)


def save_frames_by_cluster(
    rmsd_df: str, 
    initial_pred_path: str, 
    exp_pdb_dict_path: str, 
    output_dir: str, 
    frame_stride: int = 25
):

    """Saves frames from each source cluster in a pymol session. 
       
       -Frames from a source cluster are saved only if they are at least 
       frame_stride number of frames apart. 
           for instance, if frames 1-100 are present from a source
           cluster, we would only saves frames 1,11,21,etc.. 
       -Frames from a given cluster have their residues colored 
        by their respective RMSF.

    """

    unique_cluster_nums = sorted(list(set(list(rmsd_df['cluster_num']))))


    #get mapping between cluster_num and md frames  
    md_frames_info = {}
    for cluster_num in unique_cluster_nums:
        rmsd_df_subset = rmsd_df[rmsd_df['cluster_num'] == cluster_num]
        pdb_files_curr_cluster = list(rmsd_df_subset['md_frame_path'])
        md_frames_info[cluster_num] = []
        for idx,pdb_file in enumerate(pdb_files_curr_cluster):
            curr_fname = pdb_file[pdb_file.rindex('/')+1:]
            curr_frame_num = int((curr_fname.split('-')[0]).replace('frame',''))
            if (curr_frame_num % frame_stride == 0) or (idx == (len(pdb_files_curr_cluster)-1)):
                md_frames_info[cluster_num].append((curr_frame_num,pdb_file))
   
    exp_pdb_dict = {} 
    if exp_pdb_dict_path is not None:
        with open(exp_pdb_dict_path, 'rb') as f:
            exp_pdb_dict = pickle.load(f)

    exp_pdb_colors = ["paleyellow", "lightpink", "bluewhite"] 

    for i,exp_pdb_id in enumerate(exp_pdb_dict.keys()):
        print(exp_pdb_id)
        cmd.load(exp_pdb_dict[exp_pdb_id], exp_pdb_id)
        cmd.color(exp_pdb_colors[i], exp_pdb_id, 0)

    cmd.load(initial_pred_path, 'initial_AF_pred')
    cmd.color("palecyan", 'initial_AF_pred', 0)
    cmd.set('cartoon_transparency', 0.25, 'initial_AF_pred')
          
    for cluster_num in unique_cluster_nums:
        print('on cluster %s' % cluster_num)
        pdb_files_curr_cluster = md_frames_info[cluster_num]
        print('%d files for current cluster' % len(pdb_files_curr_cluster))
        for i in range(0,len(pdb_files_curr_cluster)):
            if (i+1) % 10 == 0:
                print('loading file %d (%s)' % (i+1,pdb_files_curr_cluster[i][-1]))
            curr_frame_num = pdb_files_curr_cluster[i][0]
            pdb_file = pdb_files_curr_cluster[i][-1]
            curr_fname = pdb_file[pdb_file.rindex('/')+1:]
            object_name = cluster_num
            cmd.load(pdb_file, object_name)
    if exp_pdb_dict == {}:
        selection = "all and not initial_AF_pred"
    else:
        selection = "all and not (initial_AF_pred"
        for exp_pdb_id in exp_pdb_dict:
            selection += " or %s" % exp_pdb_id
        selection += ")"
    percentile_97 = get_b_factor_percentile(selection, percentile=97)
    print(f"97th percentile of B-factors: {percentile_97}")
    cmd.show_as('cartoon', selection)
    cmd.cartoon('putty', selection)
    cmd.set("cartoon_putty_scale_min", 0, selection)
    cmd.set("cartoon_putty_scale_max", percentile_97, selection)
    cmd.set("cartoon_putty_transform", 0, selection)
    cmd.set("cartoon_putty_radius", 0.1, selection)
    cmd.spectrum("b", "rainbow", selection)
    cmd.ramp_new("rmsf_bar", cmd.get_object_list(selection)[0], [0, percentile_97], "rainbow")
    cmd.set("specular", 0)
    cmd.bg_color("white")
    pymol_session_save_dir = '%s/pymol_sessions' % output_dir
    os.makedirs(pymol_session_save_dir, exist_ok=True)
    session_name = '%s/cluster_summary.pse.gz' % pymol_session_save_dir
    print('saving %s' % session_name)
    cmd.save(session_name)
    cmd.reinitialize()



def save_frames_by_macrostate(
    rmsd_df: str, 
    initial_pred_path: str, 
    exp_pdb_dict_path: str, 
    output_dir: str, 
    frame_stride: int = 10,
    save_publication_session: bool = False,
    save_raw_pdb_files: bool = False
):

    """Saves relevant frames from a given macrostate in a pymol session. 
       
       -Specifically, for each macrostate, a pymol session is created that 
       contains frames grouped from each source cluster. 
       -Frames from a source cluster are saved only if they are at least 
       frame_stride number of frames apart. 
           for instance, if frames 1-100 are present from a source
           cluster, we would only saves frames 1,11,21,etc.. 
       -Frames from a given cluster have their residues colored 
        by their respective RMSF.

    """

    unique_macrostates = sorted(list(set(list(rmsd_df['macrostate']))))

    #corresponds to microstate that is not assigned a macrostate 
    if -1 in unique_macrostates:
        unique_macrostates.remove(-1)

    #get mapping between macrostate-cluster_num and md frames  
    md_frames_info = {}
    for macrostate in unique_macrostates:
        rmsd_df_subset = rmsd_df[rmsd_df['macrostate'] == macrostate]
        pdb_files_curr_macrostate = list(rmsd_df_subset['md_frame_path'])
        cluster_nums_curr_macrostate = list(rmsd_df_subset['cluster_num'])
        md_frames_info[macrostate] = {}
        for idx,pdb_file in enumerate(pdb_files_curr_macrostate):
            curr_fname = pdb_file[pdb_file.rindex('/')+1:]
            curr_cluster_num = cluster_nums_curr_macrostate[idx]
            if curr_cluster_num not in md_frames_info[macrostate]:
                md_frames_info[macrostate][curr_cluster_num] = []
            curr_frame_num = int((curr_fname.split('-')[0]).replace('frame',''))
            md_frames_info[macrostate][curr_cluster_num].append((curr_frame_num,curr_cluster_num,pdb_file))
   
    #get mapping between macrostate and relevant subset of md frames  
    md_frames_subset_info = {}  
    for macrostate in md_frames_info:
        md_frames_subset_info[macrostate] = [] 
        for cluster_num in md_frames_info[macrostate]:
            md_frames_info[macrostate][cluster_num] = sorted(md_frames_info[macrostate][cluster_num], key=lambda x:x[0])
            all_md_frames = md_frames_info[macrostate][cluster_num]
            num_md_frames = len(all_md_frames)
            idx_to_keep = [0]
            curr_frame = all_md_frames[0][0]
            if len(all_md_frames) == 1:
                continue
            for i in range(1,len(all_md_frames)):
                if (all_md_frames[i][0] - curr_frame) < frame_stride:
                    pass 
                else:
                    curr_frame = all_md_frames[i][0]
                    idx_to_keep.append(i)
            md_frames_subset_info[macrostate].extend([all_md_frames[i] for i in idx_to_keep])

    exp_pdb_dict = {} 
    if exp_pdb_dict_path is not None:
        with open(exp_pdb_dict_path, 'rb') as f:
            exp_pdb_dict = pickle.load(f)

    exp_pdb_colors = ["paleyellow", "lightpink", "bluewhite"] 

    for macrostate in unique_macrostates:
        cmd.load(initial_pred_path, 'initial_AF_pred')
        cmd.color("palecyan", 'initial_AF_pred', 0)
        cmd.set('cartoon_transparency', 0.25, 'initial_AF_pred')
        for i,exp_pdb_id in enumerate(exp_pdb_dict.keys()):
            cmd.load(exp_pdb_dict[exp_pdb_id], exp_pdb_id)
            cmd.color(exp_pdb_colors[i], exp_pdb_id, 0)
        print('on macrostate %d' % macrostate)
        pdb_files_curr_macrostate = md_frames_subset_info[macrostate]
        print('%d files for current macrostate' % len(pdb_files_curr_macrostate))
        for i in range(0,len(pdb_files_curr_macrostate)):
            if (i+1) % 10 == 0:
                print('loading file %d (%s)' % (i+1,pdb_files_curr_macrostate[i][-1]))
            curr_frame_num = pdb_files_curr_macrostate[i][0]
            curr_cluster_num = pdb_files_curr_macrostate[i][1]
            pdb_file = pdb_files_curr_macrostate[i][-1]
            curr_fname = pdb_file[pdb_file.rindex('/')+1:]
            object_name = curr_cluster_num
            cmd.load(pdb_file, object_name)
        if exp_pdb_dict == {}:
            selection = "all and not initial_AF_pred"
        else:
            selection = "all and not (initial_AF_pred"
            for exp_pdb_id in exp_pdb_dict:
                selection += " or %s" % exp_pdb_id
            selection += ")"
        percentile_97 = get_b_factor_percentile(selection, percentile=97)
        print(f"97th percentile of B-factors: {percentile_97}")
        cmd.show_as('cartoon', selection)
        cmd.cartoon('putty', selection)
        cmd.set("cartoon_putty_scale_min", 0, selection)
        cmd.set("cartoon_putty_scale_max", percentile_97, selection)
        cmd.set("cartoon_putty_transform", 0, selection)
        cmd.set("cartoon_putty_radius", 0.1, selection)
        cmd.spectrum("b", "rainbow", selection)
        cmd.ramp_new("rmsf_bar", cmd.get_object_list(selection)[0], [0, percentile_97], "rainbow")
        cmd.set("specular", 0)
        cmd.bg_color("white")
        pymol_session_save_dir = '%s/pymol_sessions' % output_dir
        os.makedirs(pymol_session_save_dir, exist_ok=True)
        session_name = '%s/macrostate%d.pse.gz' % (pymol_session_save_dir, macrostate)
        print('saving %s' % session_name)
        cmd.save(session_name)
        cmd.reinitialize()
        if save_raw_pdb_files:
            pdb_save_dir = '%s/pdb_files/macrostate%d' % (output_dir, macrostate)
            os.makedirs(pdb_save_dir, exist_ok=True)
            for pdb_file in pdb_files_curr_macrostate:
                curr_fname = pdb_file[pdb_file.rindex('/')+1:]
                curr_cluster_num = cluster_nums_curr_macrostate[idx]
                pdb_destination_path = '%s/%s-%s' % (pdb_save_dir, curr_cluster_num, curr_fname)
                shutil.copyfile(pdb_file, pdb_destination_path)


    if not(save_publication_session):
        return 

    #saving pymol session where structures are shown in ribbon representation 
    #with different color schemes (used to generated publication figures)

    for macrostate in unique_macrostates:
        cmd.load(initial_pred_path, 'initial_AF_pred')
        cmd.color("blue", f"initial_AF_pred and b > 90")
        cmd.color("cyan", f"initial_AF_pred and b < 90 and b > 70")
        cmd.color("yellow", f"initial_AF_pred and b < 70 and b > 50")
        cmd.color("orange", f"initial_AF_pred and b < 50")
        for i,exp_pdb_id in enumerate(exp_pdb_dict.keys()):
            cmd.load(exp_pdb_dict[exp_pdb_id], exp_pdb_id)
            cmd.color("gray80", exp_pdb_id, 0)
        print('on macrostate %d' % macrostate)
        pdb_files_curr_macrostate = md_frames_subset_info[macrostate]
        print('%d files for current macrostate' % len(pdb_files_curr_macrostate))
        for i in range(0,len(pdb_files_curr_macrostate)):
            if (i+1) % 10 == 0:
                print('loading file %d (%s)' % (i+1,pdb_files_curr_macrostate[i][-1]))
            curr_frame_num = pdb_files_curr_macrostate[i][0]
            curr_cluster_num = pdb_files_curr_macrostate[i][1]
            pdb_file = pdb_files_curr_macrostate[i][-1]
            curr_fname = pdb_file[pdb_file.rindex('/')+1:]
            object_name = curr_cluster_num
            cmd.load(pdb_file, object_name)
        cmd.show_as("ribbon", "all")
        cmd.set("ribbon_radius", 0.5)
        cmd.set("ribbon_as_cylinders", 1)
        if exp_pdb_dict == {}:
            selection = "all and not initial_AF_pred"
        else:
            selection = "all and not (initial_AF_pred"
            for exp_pdb_id in exp_pdb_dict:
                selection += " or %s" % exp_pdb_id
            selection += ")"
        percentile_97 = get_b_factor_percentile(selection, percentile=97)
        print(f"97th percentile of B-factors: {percentile_97}")
        cmd.spectrum("b", "rainbow", selection)
        cmd.ramp_new("rmsf_bar", cmd.get_object_list(selection)[0], [0, percentile_97], "rainbow")
        cmd.set('specular', 0)
        cmd.bg_color("white")
        pymol_session_save_dir = '%s/pymol_sessions-publication' % output_dir
        os.makedirs(pymol_session_save_dir, exist_ok=True)
        session_name = '%s/macrostate%d.pse.gz' % (pymol_session_save_dir, macrostate)
        print('saving %s' % session_name)
        cmd.save(session_name)
        cmd.reinitialize()

                    

def gen_output(
    linkage_matrix_path,
    microstate_info_path,
    rmsd_info_path,
    exp_pdb_dict_path,
    output_dir, 
    color_threshold,
    lag_steps,
    timestep_ns,
    num_frames_per_traj,
    min_population_threshold,
    qmin_threshold,
    hide_labels,
    save_publication_session,
):


    print(asterisk_line)
    print('SAVING FRAMES BY CLUSTER')
    rmsd_df = pd.read_csv(rmsd_info_path)
    initial_pred_path = rmsd_df['initial_pred_path'][0]
    save_frames_by_cluster(rmsd_df, initial_pred_path, exp_pdb_dict_path, output_dir) 
    print(asterisk_line)

    #get population info 
    traj = mh.opentxt(microstate_info_path)
    microstates, counts = np.unique(traj, return_counts=True)
    microstates_rel_population_dict = {} 
    for i in range(0,len(microstates)):
        microstates_rel_population_dict[microstates[i]] = round(counts[i] / np.sum(counts),3)

    print('MICROSTATES:')
    print(microstates)
    print('MICROSTATES RELATIVE POPULATION:')
    print(microstates_rel_population_dict)

    # load transitions and sort them
    transitions = np.loadtxt(linkage_matrix_path)
    if transitions.ndim == 1:
        transitions = transitions.reshape(1, -1)

    linkage_matrix_info_dict, connected_components_rel_population_dict = _transitions_to_linkage(transitions, microstates_rel_population_dict, qmin=0)

    for curr_component in linkage_matrix_info_dict:
        print(asterisk_line)
        print('on component %d' % curr_component)
        print(asterisk_line)
        curr_component_output_dir = '%s/component=%d' % (output_dir, curr_component)
        os.makedirs(curr_component_output_dir, exist_ok=True)
        gen_output_per_component(linkage_matrix_info_dict,
                   microstates_rel_population_dict,
                   connected_components_rel_population_dict,
                   curr_component,
                   linkage_matrix_path,
                   microstate_info_path,
                   rmsd_info_path,
                   exp_pdb_dict_path,
                   curr_component_output_dir,
                   color_threshold,
                   lag_steps,
                   timestep_ns,
                   num_frames_per_traj,
                   min_population_threshold,
                   qmin_threshold,
                   hide_labels,
                   save_publication_session)              


def gen_output_per_component(
    linkage_matrix_info_dict, 
    microstates_rel_population_dict,
    connected_components_rel_population_dict, 
    component_num,
    linkage_matrix_path,
    microstate_info_path,
    rmsd_info_path,
    exp_pdb_dict_path,
    output_dir, 
    color_threshold,
    lag_steps,
    timestep_ns,
    num_frames_per_traj,
    min_population_threshold,
    qmin_threshold,
    hide_labels,
    save_publication_session,
):

    linkage_matrix_fname = linkage_matrix_path[linkage_matrix_path.rindex('/')+1:]
    microstate_info_fname = microstate_info_path[microstate_info_path.rindex('/')+1:]

    traj = mh.opentxt(microstate_info_path)
    microstates, _ = np.unique(traj, return_counts=True)
    transitions = np.loadtxt(linkage_matrix_path)
    if transitions.ndim == 1:
        transitions = transitions.reshape(1, -1)

    # setup matplotlib
    pplt.use_style(figsize=2.6, figratio='golden', true_black=True)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans' 
    plt.rcParams['text.usetex'] = False

    linkage_mat = linkage_matrix_info_dict[component_num]['linkage_mat']
    states_idx_to_microstates = linkage_matrix_info_dict[component_num]['states_idx_to_microstates']
    states_idx_to_rootstates = linkage_matrix_info_dict[component_num]['states_idx_to_rootstates']
    labels = linkage_matrix_info_dict[component_num]['labels'] 

    total_rel_population = round(connected_components_rel_population_dict[component_num],2)
    print('TOTAL RELATIVE POPULATION IN CURRENT COMPONENT: %.2f' % total_rel_population)

    # get states
    nstates = len(linkage_mat) + 1
    #print(nstates)
    #print('LINKAGE MATRIX:')
    #print(linkage_mat)
    states = np.unique(linkage_mat[:, :2].astype(int))

    # replace state names by their indices
    transitions_idx, states_idx = mh.rename_by_index(
        transitions[:, :2].astype(int),
        return_permutation=True,
    )
    transitions[:, :2] = transitions_idx

    #cummulative population tree 
    state_idx_cumm_population_dict = {}
    for idx_state in states_idx_to_microstates:
        state_idx_cumm_population_dict[idx_state] = 0 
        curr_state = states_idx_to_microstates[idx_state]
        for m in curr_state:
            state_idx_cumm_population_dict[idx_state] += microstates_rel_population_dict[m]

    # use population as edge widths
    edge_widths = {
        state: EDGE_WEIGHT_SCALING_FACTOR * state_idx_cumm_population_dict[state] for state in state_idx_cumm_population_dict 
    }

    # find optimal cut
    microstates, macrostates, macrostates_assignment, microstates_to_delete = mpp_plus_cut(
        states_idx_to_rootstates=states_idx_to_rootstates,
        states_idx_to_microstates=states_idx_to_microstates,
        linkage_mat=linkage_mat,
        microstates=microstates,
        cumm_pops=state_idx_cumm_population_dict,
        microstate_pops=microstates_rel_population_dict,
        min_population_threshold=min_population_threshold,
        qmin_threshold=qmin_threshold,
    )
    n_macrostates = len(macrostates_assignment)

    rmsd_df = pd.read_csv(rmsd_info_path)
    initial_pred_path = rmsd_df['initial_pred_path'][0]
    rmsd_vals = np.array(rmsd_df['rmsd_wrt_initial_pred'])
    
    print('MICROSTATES:')
    print(microstates)
    print('MACROSTATES:')
    print(macrostates)

    rmsd_state = {
        idx_state: _mean_val_per_state(
            states_idx_to_microstates[idx_state],
            rmsd_vals,
            traj,
        )
        for idx_state in states
    }

    print('RMSD INFO:')
    print(rmsd_state)

    # define colors
    min_rmsd_color = 0
    max_rmsd_color = min(max(rmsd_state.values()),20)
    colors = {
        idx_state: _color_by_observable(rmsd_state[idx_state], min_rmsd_color, max_rmsd_color)
        for idx_state in states
    }
    # add global value

    colors[2 * (nstates - 1)] = _color_by_observable(max_rmsd_color, min_rmsd_color, max_rmsd_color)

    fig, (ax, ax_mat) = plt.subplots(
        2,
        1,
        gridspec_kw={
            'hspace': 0.05 if hide_labels else 0.3,
            'height_ratios': [9, 1],
        },
    )
    # hide spines of lower mat
    for key, spine in ax_mat.spines.items():
        spine.set_visible(False)

    dendrogram_dict = _dendrogram(
        ax=ax,
        linkage_mat=linkage_mat,
        colors=colors,
        threshold=color_threshold,
        labels=labels,
        qmin=0,
        edge_widths=edge_widths,
    )

    # plot legend
    cmap, bins = _color_by_observable(None, min_rmsd_color, max_rmsd_color)
    norm = Normalize(bins[0], bins[-1])
    label = r'$\langle RMSD \rangle_{\text{state}}$'

    cmappable = ScalarMappable(norm, cmap)
    plt.sca(ax)
    pplt.colorbar(cmappable, width='5%', label=label, position='right')

    yticks = np.arange(0.5, 1.5 + n_macrostates)
    
    xticks = 10 * np.arange(0, nstates + 1)
    cmap = LinearSegmentedColormap.from_list(
        'binary', [(0, 0, 0, 0), (0, 0, 0, 1)],
    )

    # permute macrostate assignment and label them
    macrostates_assignment = macrostates_assignment.T[
        dendrogram_dict['leaves']
    ].T
    macrostates = macrostates[dendrogram_dict['leaves']]
    microstates = microstates[dendrogram_dict['leaves']]

    # apply dynamical correction of minor branches
    dyn_corr_macrostates = mpp_plus_dyn_cor(
        macrostates=macrostates,
        microstates=microstates,
        n_macrostates=n_macrostates,
        cumm_pops=state_idx_cumm_population_dict,
        traj=traj,
        lag_steps=lag_steps,
    )

    #in traj, replaces microstates with dyn_corr_macrostates
    microstates_w_deletion = np.append(microstates, microstates_to_delete)
    dyn_corr_macrostates_w_deletion = np.append(dyn_corr_macrostates,[-1]*len(microstates_to_delete))

    macrotraj = mh.shift_data(traj, microstates_w_deletion, dyn_corr_macrostates_w_deletion)

    tmp = [_mean_val_per_state([state], rmsd_vals, macrotraj) for state in np.unique(dyn_corr_macrostates)]
    
    macrostates_sorted_by_rmsd = [
        _mean_val_per_state([state], rmsd_vals, macrotraj)
        for state in np.unique(dyn_corr_macrostates)
    ]

    macroperm = np.unique(dyn_corr_macrostates)[np.argsort(macrostates_sorted_by_rmsd)]
    dyn_corr_macrostates = mh.shift_data(
        dyn_corr_macrostates, macroperm, np.unique(dyn_corr_macrostates),
    )

    macrostates_output_path = f'{output_dir}/{linkage_matrix_fname}.macrostates'
    mh.savetxt(
        macrostates_output_path,
        np.array([microstates, dyn_corr_macrostates]).T,
        header='microstates macrostates',
        fmt='%.0f',
    )

    microstates_w_deletion = np.append(microstates, microstates_to_delete)
    dyn_corr_macrostates_w_deletion = np.append(dyn_corr_macrostates,[-1]*len(microstates_to_delete))
 
    macrotraj = mh.shift_data(traj, microstates_w_deletion, dyn_corr_macrostates_w_deletion)
    
    rmsd_df['macrostate'] = macrotraj
    rmsd_df_save_path = f'{output_dir}/rmsd_w_macrostate_info_df.csv'  
    print('SAVING %s' % rmsd_df_save_path) 
    rmsd_df.to_csv(rmsd_df_save_path, index=False)

    print(asterisk_line)
    print('SAVING FRAMES BY MACROSTATE')
    save_frames_by_macrostate(rmsd_df, initial_pred_path, exp_pdb_dict_path, output_dir, save_publication_session=save_publication_session) 
    print(asterisk_line)

    macrotraj_output_path = f'{output_dir}/{microstate_info_fname}.macrotraj'
    mh.savetxt(
        macrotraj_output_path,
        macrotraj,
        header='macrostates',
        fmt='%.0f',
    )

    num_macrostates = len(np.unique(dyn_corr_macrostates))
    num_cols = int(np.ceil(num_macrostates/2))
    num_rows = int(np.ceil(num_macrostates/num_cols))
 
    # recalculate macrostates_assignment
    for idx, mstate in enumerate(np.unique(dyn_corr_macrostates)):
        macrostates_assignment[idx] = dyn_corr_macrostates == mstate
   
    #calculates midpoint of each tick  
    xvals = 0.5 * (xticks[:-1] + xticks[1:])

    for idx, assignment in enumerate(macrostates_assignment):

        macrostates_idx = np.where(assignment == 1)[0]
        xmean = np.median(xvals[assignment == 1])

        pplt.text(
            xmean,
            yticks[idx] - (yticks[1] - yticks[0]),
            #f'{idx + 1:.0f}',
            dyn_corr_macrostates[macrostates_idx][0],
            ax=ax_mat,
            va='top',
            contour=True,
            size='small',
        )
 
    ax_mat.pcolormesh(
        xticks,
        yticks,
        macrostates_assignment,
        snap=True,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    if n_macrostates < 5:
        label_padding=15
    elif n_macrostates < 10:
        label_padding=10
    else:
        label_padding=5

    fig.suptitle('Total Relative Population: %.2f' % total_rel_population)
    ax_mat.set_yticks(yticks)
    ax_mat.set_yticklabels([])
    ax_mat.grid(visible=True, axis='y', ls='-', lw=0.5)
    ax_mat.tick_params(axis='y', length=0, width=0)
    ax_mat.set_xlim(ax.get_xlim())
    ax.set_xlabel('')
    ax_mat.set_xlabel('Macrostate', labelpad=label_padding)
    ax_mat.set_ylabel('')
    fig.align_ylabels([ax, ax_mat])

    ax_mat.set_xticks(np.arange(0.5, 0.5 + len(states)))

    #ax_mat.grid(visible=False, axis='y')
    #ax_mat.grid(visible=False, axis='x')

    if hide_labels:
        for axes in (ax,ax_mat):  # if statemat_file else [ax]:
            axes.set_xticks([])
            axes.set_xticks([], minor=True)
            axes.set_xticklabels([])
            axes.set_xticklabels([], minor=True)

    dendrogram_output_fname = 'dendrogram_component%d' % component_num
    dendrogram_output_path = f'{output_dir}/{dendrogram_output_fname}.pdf'
    print(f'saving {dendrogram_output_path}')
    pplt.savefig(f'{dendrogram_output_path}', bbox_inches='tight')
    plt.rcdefaults()
    plt.clf()
    plt.close()


    lagtimes = [lag_steps, lag_steps*2, lag_steps*5, lag_steps*10]
    tmax = num_frames_per_traj 
    ck_fig_output_path = f'{output_dir}/ck_test_plot.png'
    plt.rcParams["figure.figsize"] = (2,1)
    try:
        ck = mh.msm.ck_test(macrotraj.astype(int), lagtimes, tmax)
        mh.plot.plot_ck_test(ck=ck, frames_per_unit=int(1/timestep_ns), unit='ns', grid=(num_rows, num_cols))
        print('saving %s' % ck_fig_output_path)
        pplt.savefig(ck_fig_output_path, bbox_inches='tight', dpi=300)
    except IndexError:
        print('WARNING: unable to generate %s' % ck_fig_output_path) 



def _color_by_observable(observable, omin, omax, steps=10):
    cmap = plt.get_cmap('plasma_r', steps)
    colors = [cmap(idx) for idx in range(cmap.N)]

    bins = np.linspace(
        omin, omax, steps + 1,
    )

    if observable is None:
        return cmap, bins

    for color, rlower, rhigher in zip(colors, bins[:-1], bins[1:]):
        if rlower <= observable <= rhigher:
            return color
    
    if observable > max(bins):
        return colors[-1]
    if observable < min(bins):
        return colors[0]

    return 'k'


def _mean_val_per_state(states, observable, traj):
    if len(states) >= 1:
        mask = np.full(observable.shape[0], False)
        for state in states:
            mask = np.logical_or(
                mask,
                traj == state,
            )
        observable = observable[mask]

    return round(np.mean(observable),2)




def _transitions_to_linkage(
    trans: np.ndarray, 
    microstates_rel_population_dict: Mapping[str, float], 
    qmin: float = 0.0
):
    """Converts transition matrix to linkage matrix. Because a transition matrix 
       can have multiple connected components, we output a dictionary mapping
       each connected component to its respective linkage matrix.  
       
       Parameters
       ----------
       transitions: ndarray of shape (nstates - 1, 3)
          Three column: merged state, remaining state, qmin lebel.
       qmin: float [0, 1]
          Qmin cut-off. Returns only sublinkage-matrix.

    """

    transitions = np.copy(trans)
    states = np.unique(transitions[:, :2].astype(int))

    # sort by merging qmin level
    transitions = transitions[
        np.argsort(transitions[:, 2])
    ]

    connectivity_matrix = np.zeros((transitions.shape[0], transitions.shape[0]), dtype=bool) 
    for i in range(0,transitions.shape[0]):
        connectivity_matrix[i] = np.any(
            (transitions[i, 0] == transitions) | (transitions[i, 1] == transitions), 
            axis=1
        )
        connectivity_matrix[i, i] = False

    connectivity_matrix = connectivity_matrix.astype(int)
    graph = csr_matrix(connectivity_matrix)
    n_components, component_labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    connected_components_rel_population_dict = {} 
    for curr_component in component_labels:
        connected_components_rel_population_dict[curr_component] = 0 
        transitions_curr_component = transitions[np.where(component_labels == curr_component)]
        unique_microstates = np.unique(transitions_curr_component[:,:2])
        for m in unique_microstates:
            connected_components_rel_population_dict[curr_component] += microstates_rel_population_dict[m]

    linkage_matrix_info_dict = {}

    for curr_component in component_labels:

        linkage_matrix_info_dict[curr_component] = {} 

        transitions_curr_component = transitions[np.where(component_labels == curr_component)]
        
        # create linkage matrix
        mask_qmin = transitions_curr_component[:, 2] > qmin
        
        nstates_qmin = np.count_nonzero(mask_qmin) + 1
        linkage_mat = np.zeros((nstates_qmin - 1, 4))
 
        # replace state names by their indices
        transitions_idx, states_idx = mh.rename_by_index(
            transitions_curr_component[:, :2][mask_qmin].astype(int),
            return_permutation=True,
        )
        transitions_curr_component[:, :2][mask_qmin] = transitions_idx
        linkage_mat[:, :3] = transitions_curr_component[mask_qmin]

        # holds for each state (index) a list corresponding to the microstates
        # it consist of.
        states_idx_to_microstates = {
            idx: [
                state,
                *transitions_curr_component[~mask_qmin][:, 0][
                    transitions_curr_component[~mask_qmin][:, 1] == state
                ].astype(int),
            ]
            for idx, state in enumerate(states_idx)
        }
        states_idx_to_rootstates = {
            idx: [idx]
            for idx, _ in enumerate(states_idx)
        }

        for idx, nextstate in enumerate(range(nstates_qmin, 2 * nstates_qmin - 1)):
            statefrom, stateto = linkage_mat[idx, :2].astype(int)
            states_idx_to_microstates[nextstate] = [
                *states_idx_to_microstates[stateto],
                *states_idx_to_microstates[statefrom],
            ]
            states_idx_to_rootstates[nextstate] = [
                *states_idx_to_rootstates[stateto],
                *states_idx_to_rootstates[statefrom],
            ]

            states = linkage_mat[idx, :2].astype(int)
            for state in states:
                linkage_mat[idx + 1:, :2][
                    linkage_mat[idx + 1:, :2] == state
                ] = nextstate

        labels = [
            states_idx_to_microstates[idx][0]
            for idx in range(nstates_qmin)
        ]

        linkage_matrix_info_dict[curr_component]['linkage_mat'] = linkage_mat
        linkage_matrix_info_dict[curr_component]['states_idx_to_microstates'] = states_idx_to_microstates
        linkage_matrix_info_dict[curr_component]['states_idx_to_rootstates'] = states_idx_to_rootstates
        linkage_matrix_info_dict[curr_component]['labels'] = labels
    

    return linkage_matrix_info_dict, connected_components_rel_population_dict


def _dendrogram(
    *, ax, linkage_mat, colors, threshold, labels, qmin, edge_widths,
):
    #nstates = len(linkage_mat) + 1

    # convert color dictionary to array
    colors_arr = np.array(
        [
            to_hex(colors[state]) for state in colors 
        ],
        dtype='<U7',
    )

    dendrogram_dict = dendrogram(
        linkage_mat,
        leaf_rotation=90,
        get_leaves=True,
        color_threshold=1,
        link_color_func=lambda state_idx: colors_arr[state_idx],
        no_plot=True,
    )

    _plot_dendrogram(
        icoords=dendrogram_dict['icoord'],
        dcoords=dendrogram_dict['dcoord'],
        ivl=dendrogram_dict['ivl'],
        color_list=dendrogram_dict['color_list'],
        threshold=threshold,
        ax=ax,
        colors=colors_arr,
        labels=labels,
        qmin=qmin,
        edge_widths=edge_widths,
    )

    ax.set_ylabel('Metastability Qmin')
    ax.set_xlabel('Microstate')
    ax.grid(visible=False, axis='x')

    return dendrogram_dict

'''
def _show_xlabels(*, ax, states_perm):
    """Show the xticks together with the corresponding state names."""
    # undo changes of scipy dendrogram
    xticks = ax.get_xticks()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for line in ax.get_xticklines():
        line.set_visible(True)

    for is_major, length_scale in ((True, 4), (False, 1)):
        ax.tick_params(
            axis='x',
            length=length_scale * plt.rcParams['xtick.major.size'],
            labelrotation=90,
            pad=2,
            labelsize='xx-small',
            width=plt.rcParams['xtick.major.width'],
            which='major' if is_major else 'minor',
            top=False,
        )
        offset = 0 if is_major else 1
        ax.set_xticks(xticks[offset::2], minor=not is_major)
        ax.set_xticklabels(states_perm[offset::2], minor=not is_major)
'''

def _plot_dendrogram(
    *,
    icoords,
    dcoords,
    ivl,
    color_list,
    threshold,
    ax,
    colors,
    labels,
    qmin,
    edge_widths,
):
    """Plot dendrogram with colors at merging points."""
    threshold_color = to_hex('pplt:grey')
    # Independent variable plot width
    ivw = len(ivl) * 10
    # Dependent variable plot height
    dvw = 1.05

    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)

    ax.set_ylim([qmin, dvw])
    ax.set_xlim([-0.005 * ivw, 1.005 * ivw])
    ax.set_xticks(iv_ticks)

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(np.asarray(labels)[np.asarray(ivl).astype(int)])

    get_ancestor = _get_ancestor_func(
        icoords, dcoords, ivl,
    )

    # Let's use collections instead. This way there is a separate legend item
    # for each tree grouping, rather than stupidly one for each line segment.
    colors_used = np.unique(colors)
    color_to_lines = {color: [] for color in (*colors_used, threshold_color)}
    width_to_lines = {color: [] for color in (*colors_used, threshold_color)}
    for xline, yline, color in zip(icoords, dcoords, color_list):
        if np.max(yline) <= threshold:
            # split into left and right to color separately
            xline_l = [*xline[:2], np.mean(xline[1:3])]
            xline_r = [np.mean(xline[1:3]), *xline[2:]]

            color_l = _get_ancestor_color(
                icoords,
                dcoords,
                xline[0],
                yline[0],
                color_list,
                ivl,
                colors,
            )
            ancestors_l = get_ancestor(xline[0], yline[0])
            if len(ancestors_l) == 1:
                weight_l = np.sum([
                    edge_widths[ancestor] for ancestor in ancestors_l
                ])
                rel_pop_l_rounded = round((weight_l/EDGE_WEIGHT_SCALING_FACTOR)*100,1)
                if rel_pop_l_rounded >= 30:
                    y_disp = .03
                elif rel_pop_l_rounded >= 20:
                    y_disp = .02
                else:
                    y_disp = .01
                x_text_l = np.mean(xline_l) - 0.1
                y_text_l = np.max(yline[1:]) + y_disp
                ax.annotate(f'{rel_pop_l_rounded}%', (x_text_l, y_text_l),
                           ha='center', va='bottom', fontsize=4)
            else:
                weight_l = MIN_EDGE_WEIGHT

            color_r = _get_ancestor_color(
                icoords,
                dcoords,
                xline[3],
                yline[3],
                color_list,
                ivl,
                colors,
            )
            ancestors_r = get_ancestor(xline[3], yline[3])
            if len(ancestors_r) == 1:
                weight_r = np.sum([
                    edge_widths[ancestor] for ancestor in ancestors_r
                ])
                rel_pop_r_rounded = round((weight_r/EDGE_WEIGHT_SCALING_FACTOR)*100,1)
                if rel_pop_r_rounded >= 30:
                    y_disp = .03
                elif rel_pop_r_rounded >= 20:
                    y_disp = .02
                else:
                    y_disp = .01
                x_text_r = np.mean(xline_r) + 0.1
                y_text_r = np.max(yline[1:]) + y_disp
                ax.annotate(f'{rel_pop_r_rounded}%', (x_text_r, y_text_r),
                           ha='center', va='bottom', fontsize=4)
            else:
                weight_r = MIN_EDGE_WEIGHT


            color_to_lines[color_l].append(list(zip(xline_l, yline[:3])))
            width_to_lines[color_l].append(
                max(weight_l, MIN_EDGE_WEIGHT),
            )
            color_to_lines[color_r].append(list(zip(xline_r, yline[1:])))
            width_to_lines[color_r].append(
                max(weight_r, MIN_EDGE_WEIGHT),
            )

        elif np.min(yline) >= threshold:
            color_to_lines[threshold_color].append(list(zip(xline, yline)))
        else:
            yline_bl = [yline[0], np.max([threshold, yline[1]])]
            yline_br = [np.max([threshold, yline[2]]), yline[3]]
            color_to_lines[color].append(list(zip(xline[:2], yline_bl)))
            color_to_lines[color].append(list(zip(xline[2:], yline_br)))

            yline_thr = np.where(np.array(yline) < threshold, threshold, yline)
            color_to_lines[threshold_color].append(list(zip(xline, yline_thr)))

    # Construct the collections.
    colors_to_collections = {
        color: LineCollection(
            color_to_lines[color], colors=(color,),
            linewidths=width_to_lines[color],
        )
        for color in (*colors_used, threshold_color)
    }

    # Add all the groupings below the color threshold.
    for color in colors_used:
        ax.add_collection(colors_to_collections[color])
     #If there's a grouping of links above the color threshold, it goes last.
    ax.add_collection(colors_to_collections[threshold_color])


def _get_ancestor_color(
    xlines, ylines, xval, yval, color_list, ivl, colors,
):
    """Get the color of the ancestors."""
    # if ancestor is root
    if not yval:
        ancestor = int(ivl[int((xval - 5) // 10)])
        return colors[ancestor]

    # find ancestor color
    xy_idx = np.argwhere(
        np.logical_and(
            np.array(ylines)[:, 1] == yval,
            np.array(xlines)[:, 1:3].mean(axis=1) == xval,
        ),
    )[0][0]
    return color_list[xy_idx]


def _get_ancestor_func(
    xlines, ylines, ivl,
):
    """Get the color of the ancestors."""
    @lru_cache(maxsize=1024)
    def _get_ancestor_rec(xval, yval):
        # if ancestor is root
        if not yval:
            ancestor = int(ivl[int((xval - 5) // 10)])
            return (ancestor, )

        # find ancestor color
        xy_idx = np.argwhere(
            np.logical_and(
                np.array(ylines)[:, 1] == yval,
                np.array(xlines)[:, 1:3].mean(axis=1) == xval,
            ),
        )[0][0]
        xleft, yleft = xlines[xy_idx][0], ylines[xy_idx][0]
        xright, yright = xlines[xy_idx][3], ylines[xy_idx][3]

        return (
            *_get_ancestor_rec(xleft, yleft),
            *_get_ancestor_rec(xright, yright),
        )

    return _get_ancestor_rec


def state_sequences(macrostates, state):
    """Get continuous index sequences of macrostate in mstate assignment."""
    state_idx = np.where(macrostates == state)[0]
    idx_jump = state_idx[1:] - state_idx[:-1] != 1
    return np.array_split(
        state_idx,
        np.nonzero(idx_jump)[0] + 1,
    )


def mpp_plus_cut(
    *,
    states_idx_to_rootstates,
    states_idx_to_microstates,
    linkage_mat,
    microstates,
    cumm_pops,
    microstate_pops,
    min_population_threshold,
    qmin_threshold,
):
    """Apply MPP+ step1: Identify branches.
       This function is used to refine macrostate assignment.  
       Specifically: if a state does not have qmin above qmin_threshold OR
       a minimum population above min_population_threshold,
       then it is not treated as a macrostate.  
    """

    nstates = max(states_idx_to_rootstates.keys())
    macrostates_set = [set(states_idx_to_rootstates[nstates])]
    macrostates_leaf_set = [set(states_idx_to_microstates[nstates])]

    for state_i, state_j, qmin in reversed(linkage_mat[:, :3]):
        if cumm_pops[state_i] > min_population_threshold and cumm_pops[state_j] > min_population_threshold and qmin > qmin_threshold:
            mstate_i = set(states_idx_to_rootstates[state_i])
            macrostates_set = [
                mstate - mstate_i
                for mstate in macrostates_set
            ]
            macrostates_set.append(mstate_i)

            mstate_leaf_i = set(states_idx_to_microstates[state_i])
            macrostates_leaf_set = [
                mstate - mstate_leaf_i
                for mstate in macrostates_leaf_set
            ]
            macrostates_leaf_set.append(mstate_leaf_i)

    n_macrostates = len(macrostates_set)
    #each row corresponds to a macrostate, column represents which microstates belong to that macrostate
    macrostates_assignment = np.zeros((n_macrostates, nstates))
    for idx, mstate in enumerate(macrostates_set):
        macrostates_assignment[idx][list(mstate)] = 1
 
    macrostates = np.empty(len(microstates), dtype=np.int64)
    macrostates_population_dict = {} 
    for idx_m, macroset in enumerate(macrostates_leaf_set):
        macrostates_population_dict[idx_m + 1] = 0 

    delete_idx = []
    microstates_to_delete = []  
    for idx, microstate in enumerate(microstates):
        for idx_m, macroset in enumerate(macrostates_leaf_set):
            if microstate in macroset:
                macrostates[idx] = idx_m + 1
                macrostates_population_dict[idx_m + 1] += microstate_pops[microstate] 
                #print(f'{microstate} in macrostate {macroset}')
                break
        else:
            microstates_to_delete.append(microstate)
            delete_idx.append(idx) 
            #print(f'{microstate} not in any macrostate')

    if len(delete_idx) > 0: 
        macrostates = np.delete(macrostates, delete_idx)
        microstates = np.delete(microstates, delete_idx)

    #renmae macrostates with population info 

    return microstates, macrostates, macrostates_assignment, microstates_to_delete


def mpp_plus_dyn_cor(
    *,
    macrostates,
    microstates,
    n_macrostates,
    cumm_pops,
    traj,
    lag_steps,
):
    """Apply MPP+ step2: Dynamically correct minor branches.
       This function is used to further the refine macrostate assignment.  
    """
    # fix dynamically missassigned single-state branches
    # identify them
    dyn_corr_macrostates = macrostates[:]
    for mstate in np.unique(macrostates):
        idx_sequences = state_sequences(macrostates, mstate)
        if len(idx_sequences) > 1:
            highest_pop_sequence = np.argmax([
                np.sum([
                    cumm_pops[s] for s in microstates[seq]
                ]) for seq in idx_sequences
            ])
            idx_sequences = [
                seq for idx, seq in enumerate(idx_sequences)
                if idx != highest_pop_sequence
            ]
            for seq in idx_sequences:
                largest_state = np.max(dyn_corr_macrostates)
                for newstate, seq_idx in enumerate(
                    seq,
                    largest_state + 1,
                ):
                    dyn_corr_macrostates[seq_idx] = newstate

    # dynamically reassign all new state to previous macrostates
    mstates = np.unique(dyn_corr_macrostates)
    while len(mstates) > n_macrostates:
        tmat, mstates = mh.msm.estimate_markov_model(
            mh.shift_data(traj, microstates, dyn_corr_macrostates),
            lagtime=lag_steps,
        )

        # sort new states by increasing metastability
        qs = np.diag(tmat)[n_macrostates:]
        idx_sort = np.argsort(qs)
        newstates = mstates[n_macrostates:][idx_sort]

        deletestate = newstates[0]

        # reassign them
        idx = np.where(mstates == deletestate)[0][0]
        idxs_to = np.argsort(tmat[idx])[::-1]
        for idx_to in idxs_to:
            if idx_to == idx:
                continue
            dyn_corr_macrostates[
                dyn_corr_macrostates == deletestate
            ] = mstates[idx_to]
            break

        mstates = np.unique(dyn_corr_macrostates)

    return dyn_corr_macrostates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linkage_matrix_path", type=str, default=None, help='Path to linkage matrix created by MPP, named **_transitions.dat'
    )
    parser.add_argument(
        "--microstate_info_path", type=str, default=None, help='Path to file where each frame in trajectory is labelled with its microstate'
    )
    parser.add_argument(
        "--rmsd_info_path", type=str, default=None
    )
    parser.add_argument(
        "--exp_pdb_dict_path", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default=None 
    )
    parser.add_argument(
        "--color_threshold", type=float, default=1.0
    )
    parser.add_argument(
        "--lag_steps", type=int, default=None, help='Lagtime in frames'
    )
    parser.add_argument(
        "--timestep_ns", type=float, default=None
    )
    parser.add_argument(
        "--num_frames_per_traj", type=int, default=None
    )
    parser.add_argument(
        "--min_population_threshold", type=float, default=.005
    )
    parser.add_argument(
        "--qmin_threshold", type=float, default=.5
    )
    parser.add_argument(
        "--hide_labels", type=bool, default=True
    )
    parser.add_argument(
        "--save_publication_session", action='store_true', default=False
    )

    
    args = parser.parse_args()

    gen_output(args.linkage_matrix_path, args.microstate_info_path, args.rmsd_info_path, args.exp_pdb_dict_path, args.output_dir, args.color_threshold, args.lag_steps, args.timestep_ns, args.num_frames_per_traj, args.min_population_threshold, args.qmin_threshold, args.hide_labels, args.save_publication_session)
