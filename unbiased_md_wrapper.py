import sys
import os
import subprocess
import pandas as pd
import argparse
import glob  
import re 
import shutil 
from unbiased_md import run_sim  

asterisk_line = '******************************************************************************'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( 
        "--refined_conformations_dir", type=str, default=None, help='directory with AF-derived conformations refined via openmm' 
    )
    parser.add_argument(
        "--run_all_conformations", action='store_true', default=False
    )
    parser.add_argument(
        "--run_single_conformation_per_cluster", action='store_true', default=False
    )
    parser.add_argument(
        "--specific_cluster_to_run", type=str, default=None, help='Only run simulations corresponding to this cluster (e.g 0, 1, initial, etc.)' 
    )
    parser.add_argument(
        "--water_model", type=str, default='tip3p'
    )
    parser.add_argument(
        "--fix_pdb", action='store_true', default=False
    )
    parser.add_argument(
        "--output_dir", type=str, default=None 
    )
    parser.add_argument(
        "--production_steps", type=int, default=125000000, help='default corresponds to 125000000 (250 ns)'
    )

    args = parser.parse_args()

    if not(args.run_all_conformations) and not(args.run_single_conformation_per_cluster):
        raise ValueError("must specify whether you want to run all conformations in directory or a single conformation per cluster")

    if args.water_model not in ['tip3p', 'opc']:
        raise ValueError("water model must be one of tip3p or opc")

    heating_steps = 50000 #50 ps per temp
    equil_steps = 500000 #1 ns

    all_pdb_paths = sorted(glob.glob('%s/*openmm_refinement.pdb' % args.refined_conformations_dir))
    all_pdb_files = [f[f.rindex('/')+1:] for f in all_pdb_paths]

    print("the following %d pdb files were found in %s:" % (len(all_pdb_files), args.refined_conformations_dir))
    print(all_pdb_files)

    file_info_dict = {}

    for p in all_pdb_paths:
        f = p[p.rindex('/')+1:]
        pattern = r'cluster_(\d+).*plddt_(\d+)'
        match = re.search(pattern, f)
        if match:
            cluster_num = int(match.group(1))
            plddt_score = int(match.group(2))
            if cluster_num in file_info_dict:
                file_info_dict[cluster_num].append((plddt_score, f, p))
            else:
                file_info_dict[cluster_num] = [(plddt_score, f, p)]
        else:
            file_info_dict['initial'] = (f,p) 

    #sort by pLDDT in reverse order
    for cluster_num in file_info_dict:
        if cluster_num != 'initial':
            file_info_dict[cluster_num].sort(key=lambda x: x[0], reverse=True)        

    for cluster_num in file_info_dict:
        if cluster_num != 'initial':
            if args.specific_cluster_to_run is None or (args.specific_cluster_to_run is not None and args.specific_cluster_to_run == str(cluster_num)):
                for i in range(0,len(file_info_dict[cluster_num])):
                    input_receptor_filename = file_info_dict[cluster_num][i][1]
                    input_receptor_path = file_info_dict[cluster_num][i][2]
                    md_save_dir = '%s/cluster%d-idx%d' % (args.output_dir, cluster_num, i)
                    os.makedirs(md_save_dir, exist_ok=True)
                    dest_receptor_path = '%s/%s' % (md_save_dir, input_receptor_filename)
                    print(asterisk_line)
                    print('copying %s to %s' % (input_receptor_path, dest_receptor_path))
                    shutil.copyfile(input_receptor_path, dest_receptor_path)
                    print('RUNNING MD for %s' % dest_receptor_path)
                    run_sim(dest_receptor_path, args.fix_pdb, args.water_model, md_save_dir, heating_steps, equil_steps, args.production_steps)
                    if args.run_single_conformation_per_cluster:
                        if i == 0:
                            break 
        else:
            if args.specific_cluster_to_run is None or (args.specific_cluster_to_run is not None and args.specific_cluster_to_run == str(cluster_num)):
                input_receptor_filename = file_info_dict[cluster_num][0]
                input_receptor_path = file_info_dict[cluster_num][1]
                md_save_dir = '%s/initial' % args.output_dir
                os.makedirs(md_save_dir, exist_ok=True)
                dest_receptor_path = '%s/%s' % (md_save_dir, input_receptor_filename)
                print(asterisk_line)
                print('copying %s to %s' % (input_receptor_path, dest_receptor_path))
                print('RUNNING MD for %s' % dest_receptor_path)
                run_sim(dest_receptor_path, args.fix_pdb, args.water_model, md_save_dir, heating_steps, equil_steps, args.production_steps)
