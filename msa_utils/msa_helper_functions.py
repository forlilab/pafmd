import sys 
import pandas as pd 
from pathlib import Path

from custom_openfold_utils.pdb_utils import get_uniprot_seq     
from custom_openfold_utils.fetch_utils import fetch_pdb, fetch_pdb_metadata_df 

def list_of_strings(arg):
    return arg.split(',')

def write_fasta_file(sequence_list, unique_id_list, save_dir, output_fname) -> str:
    filename = '%s/%s.fasta' % (save_dir, output_fname)
    with open(filename, 'w') as fasta_file:
        for unique_id, sequence in zip(unique_id_list, sequence_list):
            fasta_file.write(f'>{unique_id}\n')  # Write the sequence identifier
            fasta_file.write(sequence)  # Write the sequence
            fasta_file.write('\n')
    return filename 

#msa_bath corresponds to either bfd or uniref90
def get_query_seq_from_msa(msa_path: str) -> str:   
    with open(msa_path, 'r') as f:
        msa = f.read()
    return msa.splitlines()[1]

#pdb_list does not have to include chain. in that case, chain defaults to A 
def get_pdb_w_max_seq_len(pdb_list, uniprot_id, save_pdb=False):

    uniprot_seq = get_uniprot_seq(uniprot_id)
    print('uniprot_seq: %s, length %d' % (uniprot_seq, len(uniprot_seq))) 

    longest_seq = '' 
    longest_seq_idx = -1

    max_num_pdbs_to_search = 10
    num_pdbs_to_search = min(len(pdb_list),max_num_pdbs_to_search)

    for i in range(0,num_pdbs_to_search):
        
        print('retrieving sequence for %s' % pdb_list[i])

        if len(pdb_list[i].split('_')) > 1:
            pdb_id = pdb_list[i].split('_')[0]
            chain_id = pdb_list[i].split('_')[1]
        else:
            pdb_id = pdb_list[i]
            chain_id = 'A' 

        pdb_metadata_df = fetch_pdb_metadata_df(pdb_id)
        curr_uniprot_id = pdb_metadata_df.loc[0,'uniprot_id']
        if curr_uniprot_id != uniprot_id:
            continue 
        pdb_metadata_df_relchain = pdb_metadata_df[pdb_metadata_df['chain'] == chain_id].reset_index()
        pdb_metadata_df_relchain = pdb_metadata_df_relchain[pdb_metadata_df_relchain['pdb_resnum'] != 'null']
        if len(pdb_metadata_df_relchain) == 0: 
            print("can't find pdb_chain %s" % pdb_id)
            continue 

        curr_seq = ''.join(list(pdb_metadata_df_relchain['pdb_res']))        
        print('sequence for %s: %s, length %d' % (pdb_list[i], curr_seq, len(curr_seq)))

        if len(curr_seq) > len(longest_seq) and len(curr_seq) <= len(uniprot_seq):
            longest_seq = curr_seq 
            longest_seq_idx = i 

    rel_pdb = pdb_list[longest_seq_idx]
    if len(rel_pdb.split('_')) == 1:
        rel_pdb_id_w_chain = '%s_%s' % (rel_pdb, 'A') 
    else:
        rel_pdb_id_w_chain = pdb_list[longest_seq_idx]

    print('longest seq corresponds to %s' % rel_pdb_id_w_chain)

    if save_pdb:
        pdb_save_dir = '%s/monomer' % args.pdb_save_dir
        Path(pdb_save_dir).mkdir(parents=True, exist_ok=True)
        pdb_path, pdb_seq = fetch_pdb(rel_pdb_id_w_chain, pdb_save_dir)

    return rel_pdb_id_w_chain



