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

from typing import List

import requests
import urllib.request

from pymol import cmd

from custom_openfold_utils.pdb_utils import clean_pdb, convert_mmcif_to_pdb 

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

sifts_script = '%s/parse_sifts.py' % current_dir

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

def fetch_mmcif(pdb_id: str, save_dir: str):

    """
    Args:
        pdb_id: e.g 1xyz
    """     
    pdb_list = PDBList()
    cif_fname = pdb_list.retrieve_pdb_file(pdb_id, file_format='mmCif', pdir=save_dir)
    return cif_fname


def fetch_pdb(pdb_id: str, save_dir: str, clean=False) -> str:

    """
    Args:
        pdb_id: e.g 1xyz_A or 1xyz
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
        chain_id_list = None
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

def fetch_af_pdb(uniprot_id: str, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)

    url = 'https://alphafold.ebi.ac.uk/api/prediction/%s' % uniprot_id
    params = {'key': 'AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'}
    headers = {'accept': 'application/json'}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        logger.info("Failed to retreive AF metadata")
        logger.info(response.raise_for_status())
        return None  

    af_id = data[0]['entryId']
    cif_url = data[0]['cifUrl']
    af_seq = data[0]['uniprotSequence']

    cif_path = '%s/%s.cif' % (save_dir, af_id)
    urllib.request.urlretrieve(cif_url, cif_path)
    pdb_path = cif_path.replace('.cif','.pdb')
    pdb_path = pdb_path.replace('-','')
    convert_mmcif_to_pdb(cif_path, pdb_path)
    os.remove(cif_path)    
 
    return (pdb_path, af_id, af_seq)

