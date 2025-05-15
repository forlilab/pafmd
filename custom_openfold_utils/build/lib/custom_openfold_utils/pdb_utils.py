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

from Bio.PDB import PDBList, PDBParser, MMCIFParser, MMCIF2Dict, Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from Bio.SeqUtils import seq1
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from typing import List

from custom_openfold_utils import cleanup 
from custom_openfold_utils import protein 

try:
    import openmm
    from openmm import unit
    from openmm import app as openmm_app
    from openmm.app.internal.pdbstructure import PdbStructure 
    from openmm.app import PDBFile, PDBxFile
    from openmm.app import Modeller
except ImportError:
    import simtk.openmm
    from simtk.openmm import unit
    from simtk.openmm import app as openmm_app
    from simtk.openmm.app.internal.pdbstructure import PdbStructure
    from simtk.openmm.app import PDBFile, PDBxFile
    from simtk.openmm.app import Modeller

import requests
import urllib.request

from pymol import cmd
from prody import * 

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

sifts_script = '%s/parse_sifts.py' % current_dir
TMalign_path = '../TMalign/TMalign'

##misc. functions##

def num_to_chain(number):
    if 0 <= number < 26:
        return string.ascii_uppercase[number]
    else:
        raise ValueError("Number should be between 0 and 25 for A to Z chains")


def get_model_name(pdb_path: str):
    return pdb_path[pdb_path.rfind('/')+1:pdb_path.rfind('.')]


def get_pymol_cmd_superimpose(pdb_model_name, chain=None):
    if chain is not None:
        out = '%s and chain %s' % (pdb_model_name,chain)
    else:
        out = '%s' % (pdb_model_name)
    return out


def get_pymol_cmd_save(pdb_model_name, chain=None):
    if chain is not None:
        out = '%s and chain %s' % (pdb_model_name,chain)
    else:
        out = '%s' % (pdb_model_name)
    return out 

##fetch functions##

def fetch_pdb_metadata_df(pdb_id):
    curl_cmd = ['curl --silent ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/%s.xml.gz | gunzip | python %s' % (pdb_id.lower(),sifts_script)] 
    output = subprocess.check_output(curl_cmd, shell=True, universal_newlines=True)
    lines = output.strip().split('\n')

    #SIFTS file provides info on the pdb sequence construct without coordinate info. 
    #This is reflected in the column pdb_res. 
    #If a residue is missing, the pdb_resnum will show up as null. 
    #If a mutation in a residue was made, the residue will differ from the uniprot_res.

    lines_arr = [x.split('\t') for x in lines]
    pdb_metadata_df = pd.DataFrame(lines_arr, columns = ['pdb_id', 'chain', 'pdb_res', 'pdb_resnum', 'uniprot_id', 'uniprot_res', 'uniprot_resnum'])
    pdb_metadata_df = pdb_metadata_df[['pdb_id', 'chain', 'uniprot_id', 'pdb_res', 'pdb_resnum', 'uniprot_res', 'uniprot_resnum']]
    return pdb_metadata_df

def fetch_pdb(pdb_id: str, save_dir: str, clean=False) -> str:

    """
    args:
        pdb_id: e.g 1xyz_a or 1xyz
        save_dir: e.g ./pdb_raw_structures_folder
    """ 

    pdb_save_path = "%s/%s.pdb" % (save_dir, pdb_id)

    if os.path.exists(pdb_save_path):
        return pdb_save_path

    cmd.reinitialize()
    if len(pdb_id.split('_')) > 1:
        pdb_id_wo_chain = pdb_id.split('_')[0]
        chain_id = pdb_id.split('_')[1]
        cmd.fetch(pdb_id_wo_chain, async_=0)
        cmd.save(pdb_save_path, "chain %s and %s" % (chain_id, pdb_id_wo_chain))
        chain_id_list = [chain_id]
    else:
        cmd.fetch(pdb_id, async_=0)
        cmd.save(pdb_save_path, pdb_id)
        chain_id_list = none
    cmd.delete('all')

    if len(pdb_id.split('_')) > 1:
        fetch_path = './%s.cif' % pdb_id_wo_chain
    else:
        fetch_path = './%s.cif' % pdb_id

    if os.path.exists(fetch_path):
        os.remove(fetch_path)    

    if clean:
        with open(pdb_save_path, "r") as f:
            pdb_str = f.read()
        clean_pdb(pdb_save_path, pdb_str)

    return pdb_save_path


##file format conversion functions##

def convert_mmcif_to_pdb(cif_path, pdb_path):
    cmd.reinitialize()
    cmd.load(cif_path, 'protein')
    cmd.save(pdb_path, 'protein')
    cmd.delete('all')


def convert_pdb_to_mmcif(pdb_path: str, output_dir: str) -> str:
    """
    Args:
        pdb_path: path to .pdb file 
    """ 

    os.makedirs(output_dir, exist_ok=True)
    model_name = pdb_path[pdb_path.rindex('/')+1:pdb_path.rindex('.pdb')]
    cif_path = os.path.join(output_dir, f'{model_name}.cif')

    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    prot = protein.from_pdb_string(pdb_str)
    cif_string = protein.to_modelcif(prot)
    with open(cif_path, 'w') as f:
        f.write(cif_string)

    return cif_path

def get_cif_string_from_pdb(pdb_path: str) -> str:
    """
    Args:
        pdb_path: path to .pdb file 
    """ 

    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    prot = protein.from_pdb_string(pdb_str)
    cif_string = protein.to_modelcif(prot)

    return cif_string 
 


##get functions##

def get_uniprot_seq(uniprot_id):
    base_url = "http://www.uniprot.org/uniprot"
    full_url = "%s/%s.fasta" % (base_url, uniprot_id)
    response = requests.post(full_url)
    if response.status_code == 404:
        raise ValueError("uniprot_id %s is invalid, no sequence found"  % uniprot_id)
    response_output = ''.join(response.text)
    seq = StringIO(response_output)
    parsed_seq = list(SeqIO.parse(seq, 'fasta'))[0]
    return str(parsed_seq.seq)


def get_uniprot_id(pdb_id):
    if len(pdb_id.split('_')) > 1:
        pdb_id = pdb_id.split('_')[0]
    base_url = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot"
    full_url = "%s/%s" % (base_url, pdb_id)
    response = requests.get(full_url)
    if response.status_code == 404:
        raise ValueError("pdb_id %s is invalid, no uniprot found"  % pdb_id)
    data = response.json()
    uniprot_id = list(data[pdb_id]['UniProt'].keys())[0]
    return uniprot_id


def get_pdb_id_seq(pdb_id, uniprot_id=None):
    if len(pdb_id.split('_')) > 1:
        pdb_id_copy = pdb_id
        pdb_id = pdb_id_copy.split('_')[0]
        chain_id = pdb_id_copy.split('_')[1]
    else:
        chain_id = 'A'

    pdb_metadata_df = fetch_pdb_metadata_df(pdb_id)
    if uniprot_id is not None:
        curr_uniprot_id = pdb_metadata_df.loc[0,'uniprot_id']
        if curr_uniprot_id != uniprot_id:
            raise ValueError('fetching pdb_id seq, but uniprot_id does not match expected')

    pdb_metadata_df_relchain = pdb_metadata_df[pdb_metadata_df['chain'] == chain_id].reset_index()
    pdb_metadata_df_relchain = pdb_metadata_df_relchain[pdb_metadata_df_relchain['pdb_resnum'] != 'null']
    if len(pdb_metadata_df_relchain) == 0: 
        raise ValueError("fetching pdb_id seq, but can't find pdb_id %s in SIFTS database" % pdb_id)

    seq = ''.join(list(pdb_metadata_df_relchain['pdb_res']))       
    return seq  


def get_pdb_path_seq(pdb_path: str, chain_list: List[str]):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    seq = '' 
    for model in structure:
        for chain in model:
            if chain_list is not None:
                if chain.id in chain_list:
                    for residue in chain.get_residues():
                        seq += seq1(residue.get_resname())
            else:
                for residue in chain.get_residues():
                    seq += seq1(residue.get_resname())

    return seq


def get_pdb_str(pdb_id: str, pdb_path: str) -> str:

    """
    Args:
        pdb_id: e.g 1xf2
        pdb_path: e.g ./pdb_raw_structures_folder/pdb1xf2.cif
    """ 

    logger.info('Getting pdb_str for %s,%s' % (pdb_id,pdb_path))

    if '.cif' in pdb_path:
        parser = MMCIFParser()
        structure = parser.get_structure(pdb_id, pdb_path)
        new_pdb = io.StringIO()
        pdb_io=PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(new_pdb)
        return new_pdb.getvalue()
    else:
        raise ValueError('.cif must be in %s' % pdb_path)



def get_ca_coords_matrix(pdb_path):

    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model['A']
    
    residue_idx_ca_coords_matrix = [] 

    for residue in chain:
        if 'CA' in residue:
            ca_atom = residue['CA']
            ca_coords = ca_atom.get_coord()
            residue_name = residue.get_resname()
            residue_idx = residue.get_id()[1]-1
            residue_idx_ca_coords_matrix.append([ca_coords[0], ca_coords[1], ca_coords[2]])

    return np.array(residue_idx_ca_coords_matrix)


def get_ca_coords_dict(pdb_path):
    
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model['A']
    
    residue_idx_ca_coords_dict = {} #maps residue index to ca coordinates 

    for residue in chain:
        if 'CA' in residue:
            ca_atom = residue['CA']
            ca_coords = ca_atom.get_coord()
            residue_name = residue.get_resname()
            residue_idx = residue.get_id()[1]-1
            residue_idx_ca_coords_dict[residue_idx] = (ca_coords[0], ca_coords[1], ca_coords[2], residue_name)
        '''else:
            print(residue)
            print(residue.get_id()[1]-1)
            for atom in residue:
                print(f"  Atom: {atom.get_name()} {atom.get_coord()}") # Print atom name and coordinates'''

    return residue_idx_ca_coords_dict


def get_bfactor(cif_path: str):

    structure = MMCIF2Dict.MMCIF2Dict(cif_path)
    b_factor = structure['_atom_site.B_iso_or_equiv']
    res = structure['_atom_site.label_comp_id']
    atom = structure['_atom_site.label_atom_id']

    b_factor_ca = []
    res_ca = []

    for i in range(0,len(atom)):
        if atom[i] == 'CA':
            b_factor_ca.append(float(b_factor[i]))
            res_ca.append(res[i])

    b_factor_ca = np.array(b_factor_ca)
    mean_bfactor_ca = np.mean(b_factor_ca)
    return b_factor_ca, mean_bfactor_ca


def get_af_disordered_domains(pdb_path: str):
    """pdb_path assumes AF structure. for region to be considered disordered domain, must be >=3 residues in length
    """
    #plddt_threshold sourced from https://onlinelibrary.wiley.com/doi/10.1002/pro.4466

    cif_path = convert_pdb_to_mmcif(pdb_path, './cif_temp')
    plddt_scores, mean_plddt = get_bfactor(cif_path)
    os.remove(cif_path)
    af_seq = get_pdb_path_seq(pdb_path, None)

    af_disordered = 1-plddt_scores/100
    af_disordered = af_disordered >= .31
     
    af_disordered_domains_idx = []
    start_idx = 0
    while start_idx < len(af_disordered):
        if af_disordered[start_idx]:
            end_idx = start_idx+1
            if end_idx < len(af_disordered):
                while af_disordered[end_idx]:
                    end_idx += 1
                    if end_idx >= len(af_disordered):
                        break 
            #by postcondition: af_disordered[end_idx] = False   
            seg_len = end_idx-start_idx
            if seg_len >= 3:
                af_disordered_domains_idx.append([start_idx, end_idx-1])
            start_idx = end_idx+1
        else:
            start_idx += 1 
               
    af_disordered_domains_seq = [] 
    for i in range(0,len(af_disordered_domains_idx)):
        af_disordered_domains_seq.append([af_seq[af_disordered_domains_idx[i][0]:af_disordered_domains_idx[i][1]+1]])

    return af_disordered_domains_idx, af_disordered_domains_seq

def get_af_disordered_residues(af_disordered_domains_idx: List[List[int]]):
    
    af_disordered_residues_idx = []
    for i in range(0,len(af_disordered_domains_idx)):
        d = af_disordered_domains_idx[i]
        af_start_idx = d[0]
        af_end_idx = d[1]
        for j in range(af_start_idx,af_end_idx+1):
            af_disordered_residues_idx.append(j)
    return af_disordered_residues_idx


def get_pdb_disordered_residues(af_seq: str, pdb_seq: str, af_disordered_domains_idx: List[List[int]]):

    alignments = pairwise2.align.globalxx(af_seq, pdb_seq)
    af_seq_aligned = alignments[0].seqA
    pdb_seq_aligned = alignments[0].seqB

    af_seq_original_to_aligned_idx_mapping = {} #maps each index in af_seq to the corresponding index in af_seq_aligned (i.e accounting for gaps) 
    for i in range(0,len(af_seq_aligned)):
        if af_seq_aligned[i] != '-':
            af_seq_original_to_aligned_idx_mapping[i-af_seq_aligned[0:i].count('-')] = i 

    pdb_disordered_residues_idx = [] 
    pdb_disordered_residues_seq = [] 

    #if ith residue of af_seq is disordered
    #then, to get the corresponding index in pdb_seq
    #we need to subtract num_gaps(pdb_seq_aligned[0:i]) from i 

    for i in range(0,len(af_disordered_domains_idx)):
        d = af_disordered_domains_idx[i]
        af_start_idx = d[0]
        af_end_idx = d[1]
        af_start_idx_aligned = af_seq_original_to_aligned_idx_mapping[af_start_idx]
        af_end_idx_aligned = af_seq_original_to_aligned_idx_mapping[af_end_idx]
        for j in range(af_start_idx_aligned, af_end_idx_aligned+1): 
            if pdb_seq_aligned[j] != '-':
                num_gaps = pdb_seq_aligned[0:j].count('-')
                pdb_seq_original_idx = j-num_gaps
                pdb_disordered_residues_idx.append(pdb_seq_original_idx)
                pdb_disordered_residues_seq.append(pdb_seq[pdb_seq_original_idx])

    return pdb_disordered_residues_idx, pdb_disordered_residues_seq 
   


###alignment related functions###

def get_rmsd(pdb1_path, pdb2_path, pdb1_chain=None, pdb2_chain=None):

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    cmd.reinitialize()
    cmd.load(pdb1_path)
    cmd.load(pdb2_path)

    s1 = get_pymol_cmd_superimpose(pdb1_model_name, pdb1_chain)
    s2 = get_pymol_cmd_superimpose(pdb2_model_name, pdb2_chain)

    out = cmd.super(s2,s1) #this superimposes s2 onto s1
    rmsd = out[0]*10 #convert to angstrom

    if rmsd < 0:
        logger.info("RMSD < 0")
        rmsd = 0 

    cmd.delete('all')

    return rmsd 


def align_and_get_rmsd(pdb1_path, pdb2_path, pdb1_chain=None, pdb2_chain=None):

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    cmd.reinitialize()
    cmd.load(pdb1_path)
    cmd.load(pdb2_path)

    s1 = get_pymol_cmd_superimpose(pdb1_model_name, pdb1_chain)
    s2 = get_pymol_cmd_superimpose(pdb2_model_name, pdb2_chain)

    logger.info('super %s,%s' % (s2,s1))

    out = cmd.super(s2,s1) #this superimposes s2 onto s1
    rmsd = out[0]*10 #convert to angstrom
    logger.info('RMSD: %.3f' % rmsd)

    if rmsd < 0:
        logger.info("RMSD < 0")
        rmsd = 0 

    s2 = get_pymol_cmd_save(pdb2_model_name)
    cmd.save(pdb2_path, s2)
    cmd.delete('all')

    return rmsd 


def tmalign_wrapper(pdb1_path, pdb2_path):
    
    p = subprocess.Popen(
        f'{TMalign_path} {pdb1_path} {pdb2_path} | grep -E "RMSD|TM-score=" ',
        stdout=subprocess.PIPE,
        shell=True,
    )
    
    output, __ = p.communicate()
    tm_score = float(str(output)[:-3].split("TM-score=")[-1].split("(if")[0])

    return tm_score


def superimpose_wrapper_monomer(pdb1_full_id: str, pdb2_full_id: str, pdb1_source: str, pdb2_source: str, pdb1_path: str, pdb2_path: str, save_dir: str):

    """
    Args:
        pdb1_full_id: e.g 1xf2_A
        pdb2_full_id: e.g 1ya3_B
        pdb1_source: either 'pdb' or 'pred_*'
        pdb2_source: either 'pdb' or 'pred_*' 
        pdb1_path: path to first protein structure (if None, then retrieve pdb)
        pdb2_path: path to second protein structure (if None, then retrieve pdb)
    """ 
 
    logger.info('pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))

    if 'pred' in pdb1_source:
        logger.info('ERROR: pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))
        raise ValueError('Because we are superimposing pdb2 on pdb1, pdb1 should be from PDB while pdb2 should be a predicted structure or from PDB')

    if (pdb1_source == 'pdb') and (pdb1_path is None):
        pdb1_id = pdb1_full_id.split('_')[0]
        pdb1_chain = pdb1_full_id.split('_')[1] 
    else:
        pdb1_id = None
        pdb1_chain = None

    if (pdb2_source == 'pdb') and (pdb2_path is None):
        pdb2_id = pdb2_full_id.split('_')[0]
        pdb2_chain = pdb2_full_id.split('_')[1] 
    else:
        pdb2_id = None
        pdb2_chain = None 

    if pdb1_source == 'pred' and pdb2_source == 'pdb':
        raise ValueError('Incompatible combination of pdb1_source and pdb2_source')

    pdb_ref_struc_folder = '%s/pdb_ref_structure' % save_dir
    Path(pdb_ref_struc_folder).mkdir(parents=True, exist_ok=True)
   
    pdb_superimposed_folder = '%s/pdb_superimposed_structures' % save_dir 
    Path(pdb_superimposed_folder).mkdir(parents=True, exist_ok=True)


    if (pdb1_path is None) and (pdb1_source == 'pdb'):
        pdb1_path = fetch_pdb(pdb1_full_id, pdb_ref_struc_folder)
        pdb1_chain_path = pdb1_path.replace('%s.pdb' % pdb1_id, '%s.pdb' % pdb1_full_id)
        pdb1_path = save_pdb_chain(pdb1_path, pdb1_chain_path, pdb1_chain) 
    else:
        if (pdb1_path is not None) and (pdb1_source == 'pdb'):
            shutil.copyfile(pdb1_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb1_full_id))

    if (pdb2_path is None) and (pdb2_source == 'pdb'):
        pdb2_path = fetch_pdb(pdb2_full_id, pdb_ref_struc_folder) 
        pdb2_chain_path = pdb2_path.replace('%s.pdb' % pdb2_id, '%s.pdb' % pdb2_full_id)
        pdb2_path = save_pdb_chain(pdb2_path, pdb2_chain_path, pdb2_chain) 
    else:
        if (pdb2_path is not None) and (pdb2_source == 'pdb'):
            shutil.copyfile(pdb2_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb2_full_id))

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    with open(pdb1_path, "r") as f:
        pdb1_str = f.read()
    with open(pdb2_path, "r") as f:
        pdb2_str = f.read()

    pdb1_input_path = pdb1_path
    pdb2_input_path = pdb2_path 

    pdb1_path_clean = pdb1_path.replace('.pdb', '_clean.pdb') 
    pdb2_path_clean = pdb2_path.replace('.pdb', '_clean.pdb')

    if (pdb1_path is not None) and (pdb1_source == 'pdb'): 
        logger.info('Cleaning pdb file %s' % pdb1_path)
        clean_pdb(pdb1_path_clean, pdb1_str)
        pdb1_input_path = pdb1_path_clean
    if (pdb2_path is not None) and (pdb2_source == 'pdb'):
        logger.info('Cleaning pdb file %s' % pdb2_path)
        clean_pdb(pdb2_path_clean, pdb2_str)
        pdb2_input_path = pdb2_path_clean
    
    pdb2_output_path = '%s/%s.pdb' % (pdb_superimposed_folder, pdb2_model_name)
    shutil.copyfile(pdb2_input_path, pdb2_output_path)

    #pdb1_chain and pdb2_chain are set to A because
    #clean_pdb replaces chain_id with A 
    rmsd = align_and_get_rmsd(pdb1_input_path, pdb2_output_path, 'A', 'A')
    tm_score = tmalign_wrapper(pdb1_input_path, pdb2_output_path)
    logger.info('SAVING ALIGNED PDB AT %s' % pdb2_output_path)
 
    return rmsd, tm_score, pdb1_input_path, pdb2_output_path


def superimpose_wrapper_multimer(pdb1_id: str, pdb2_id: str, pdb1_source: str, pdb2_source: str, pdb1_path: str, pdb2_path: str, save_dir: str):

    """
    Args:
        pdb1_id: e.g 1ixf2
        pdb2_id: e.g 1ya3
        pdb1_source: either 'pdb' or 'pred_*'
        pdb2_source: either 'pdb' or 'pred_*' 
        pdb1_path: path to first protein structure (if None, then retrieve pdb)
        pdb2_path: path to second protein structure (if None, then retrieve pdb)
    """ 
 
    logger.info('pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))

    if 'pred' in pdb1_source:
        logger.info('ERROR: pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))
        raise ValueError('because we are superimposing pdb2 on pdb1, pdb1 should be from PDB while pdb2 should be a predicted structure or from PDB')

    if pdb1_source == 'pred' and pdb2_source == 'pdb':
        raise ValueError('Incompatible combination of pdb1_source and pdb2_source')

    pdb_ref_struc_folder = '%s/pdb_ref_structure' % save_dir
    Path(pdb_ref_struc_folder).mkdir(parents=True, exist_ok=True)
   
    pdb_superimposed_folder = '%s/pdb_superimposed_structures' % save_dir             
    Path(pdb_superimposed_folder).mkdir(parents=True, exist_ok=True)

    if (pdb1_path is None) and (pdb1_source == 'pdb'):
        pdb1_path = fetch_pdb(pdb1_id, pdb_ref_struc_folder)
    else:
        if (pdb1_path is not None) and (pdb1_source == 'pdb'):
            shutil.copyfile(pdb1_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb1_id))

    if (pdb2_path is None) and (pdb2_source == 'pdb'):
        pdb2_path = fetch_pdb(pdb2_id, pdb_ref_struc_folder)
    else:
        if (pdb2_path is not None) and (pdb2_source == 'pdb'):
            shutil.copyfile(pdb2_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb2_id))

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    with open(pdb1_path, "r") as f:
        pdb1_str = f.read()
    with open(pdb2_path, "r") as f:
        pdb2_str = f.read()

    pdb1_input_path = pdb1_path
    pdb2_input_path = pdb2_path 

    pdb1_path_clean = pdb1_path.replace('.pdb', '_clean.pdb') 
    pdb2_path_clean = pdb2_path.replace('.pdb', '_clean.pdb')

    if (pdb1_path is not None) and (pdb1_source == 'pdb'): 
        logger.info('Cleaning pdb file %s' % pdb1_path)
        clean_pdb(pdb1_path_clean, pdb1_str)
        pdb1_input_path = pdb1_path_clean
    if (pdb2_path is not None) and (pdb2_source == 'pdb'):
        logger.info('Cleaning pdb file %s' % pdb2_path)
        clean_pdb(pdb2_path_clean, pdb2_str)
        pdb2_input_path = pdb2_path_clean
 
    pdb2_output_path = '%s/%s.pdb' % (pdb_superimposed_folder, pdb2_model_name)
    shutil.copyfile(pdb2_input_path, pdb2_output_path)

    pdb1_chain = None
    pdb2_chain = None 
    rmsd = align_and_get_rmsd(pdb1_input_path, pdb2_output_path, pdb1_chain, pdb2_chain)
    logger.info('SAVING ALIGNED PDB AT %s' % pdb2_output_path)

    return rmsd, pdb1_input_path, pdb2_output_path
 


####write functions#### 

def save_pdb_chain(pdb_input_path, pdb_output_path, chain):
    if os.path.isfile(pdb_output_path) == False:
        cmd.reinitialize()
        cmd.load(pdb_input_path)
        pdb_model_name = get_model_name(pdb_input_path)
        s = get_pymol_cmd_save(pdb_model_name, chain)
        cmd.save(pdb_output_path, s)
        cmd.delete('all')
    return pdb_output_path

def save_ca_coords(pdb_input_path, ca_coords, pdb_output_path):
    ca_atoms = parsePDB(pdb_input_path, subset='ca')
    ca_atoms.setCoords(ca_coords)
    print('saving %s' % pdb_output_path)
    writePDB(pdb_output_path, ca_atoms)

def _get_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
    """Returns a pdb string provided OpenMM topology and positions."""
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()

def clean_pdb(pdb_path: str, pdb_str: str, add_missing_residues: bool = False):
    pdb_file = io.StringIO(pdb_str)
    alterations_info = {}
    if add_missing_residues:
        fixed_pdb = cleanup.fix_pdb(pdb_file, alterations_info)
    else:
        fixed_pdb = cleanup.fix_pdb_wo_adding_missing_residues(pdb_file, alterations_info)
    fixed_pdb_file = io.StringIO(fixed_pdb)
    pdb_structure = PdbStructure(fixed_pdb_file)
    cleanup.clean_structure(pdb_structure, alterations_info)
    as_file = openmm_app.PDBFile(pdb_structure)
    pdb_str_cleaned = _get_pdb_string(as_file.getTopology(), as_file.getPositions())
    with open(pdb_path, "w") as f:
        f.write(pdb_str_cleaned)


def renumber_chain_wrt_reference(pdb_modify_path, pdb_ref_path):
    # Parse the PDB files
    parser = PDBParser()
    logger.info('Renumbering chains of %s to match %s' % (pdb_modify_path, pdb_ref_path))
    structure_to_modify = parser.get_structure('renumber', pdb_modify_path)
    reference_structure = parser.get_structure('reference', pdb_ref_path)

    reference_chains_list = [] 
    for model in reference_structure:
        for chain in model:
            reference_chains_list.append(chain.id)

    num_chains = len(reference_chains_list)
    
    chain_num = 0 
    for model in structure_to_modify:
        for chain in model:
            chain.id = reference_chains_list[chain_num]
            chain_num += 1

    # Save the modified structure
    io = PDBIO()
    io.set_structure(structure_to_modify)
    io.save(pdb_modify_path)

    return num_chains


def select_rel_chain_and_delete_residues(cif_input_path: str, cif_output_path: str, chain_id: str, residues_delete_idx: List[int]):
    
    pdb = PDBxFile(cif_input_path)
    modeller = Modeller(pdb.topology, pdb.positions)

    chains_to_delete = [] 
    for chain in modeller.getTopology().chains():
        if chain.id != chain_id:
            chains_to_delete.append(chain)
        else:
            print('keeping:')
            print(chain)
    print('deleting chains:')
    print(chains_to_delete)
    modeller.delete(chains_to_delete)    

    residues_to_delete = [] 
    for residue in modeller.topology.residues(): 
        if residue.index in residues_delete_idx:
            residues_to_delete.append(residue)
    print('deleting:')
    print(residues_to_delete)
    modeller.delete(residues_to_delete) 

    with open(cif_output_path, 'w') as f:
        PDBxFile.writeFile(modeller.topology, modeller.positions, f)


def assemble_multiple_pdbs_to_single_file(pdb_path_list, output_path_wo_extension):

    pdb_output_path = '%s.pdb' % output_path_wo_extension
    cif_output_path = '%s.cif' % output_path_wo_extension

    cmd.reinitialize()
    pdb_str_all = []
    for i,pdb_path in enumerate(pdb_path_list):
        pdb_str = 'protein_%d' % i
        cmd.load(pdb_path, pdb_str)
        cmd.alter('%s and chain A' % pdb_str, "chain='%s'" % num_to_chain(i))
        pdb_str_all.append(pdb_str)
    cmd.sort()
    pdb_or_str = ' or '.join(pdb_str_all)
    cmd.create("multimer", pdb_or_str)
    cmd.save(pdb_output_path, 'multimer')
    cmd.delete('all')

    with open(pdb_output_path, "r") as f:
        pdb_str = f.read()
    prot = protein.from_pdb_string(pdb_str)
    cif_string = protein.to_modelcif(prot)
    with open(cif_output_path, 'w') as f:
        f.write(cif_string)

    os.remove(pdb_output_path)   
    return cif_output_path 

