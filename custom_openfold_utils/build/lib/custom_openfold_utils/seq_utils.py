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

from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from typing import List

import requests
import urllib.request

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

def get_starting_idx_aligned(aligned_seq):
    #becase of issues with aligning n/c terminal regions of sequences, we want to start where there are 
    #consecutive hits
    for i in range(0,len(aligned_seq)-1):
        if aligned_seq[i] != '-':
            if aligned_seq[i+1] != '-':
                return i 
    
    return -1


def get_ending_idx_aligned(aligned_seq):
    #becase of issues with aligning n/c terminal regions of sequences, we want to start where there are 
    #consecutive hits
    for i in range(len(aligned_seq)-1,1,-1):
        if aligned_seq[i] != '-':
            if aligned_seq[i-1] != '-':
                return i 
    
    return -1

def get_common_aligned_residues_idx_excluding_flanking_regions(seq1: str, seq2: str):

    seq1_start_idx = get_starting_idx_aligned(seq1)
    seq2_start_idx = get_starting_idx_aligned(seq2)

    seq1_end_idx = get_ending_idx_aligned(seq1)
    seq2_end_idx = get_ending_idx_aligned(seq2)

    start_idx = max(seq1_start_idx, seq2_start_idx)
    end_idx = min(seq1_end_idx, seq2_end_idx)

    return start_idx, end_idx
    

def get_residues_idx_in_seq1_and_seq2(seq1: str, seq2: str):
    """
    seq1: MS-E-
    seq2: MSDER
    return: [0,1,2], [0,1,3]
    """ 

    alignments = pairwise2.align.globalxx(seq1, seq2)
    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    start_aligned_to_original_idx, end_aligned_to_original_idx = get_common_aligned_residues_idx_excluding_flanking_regions(seq1_aligned, seq2_aligned)

    seq1_residues_idx = [] 
    seq2_residues_idx = [] 

    for i in range(start_aligned_to_original_idx, end_aligned_to_original_idx+1):
        if seq1_aligned[i] != '-' and seq2_aligned[i] != '-':
            num_gaps_seq1 = seq1_aligned[0:i].count('-')
            num_gaps_seq2 = seq2_aligned[0:i].count('-')
            seq1_residues_idx.append(i-num_gaps_seq1)
            seq2_residues_idx.append(i-num_gaps_seq2)

    return seq1_residues_idx, seq2_residues_idx


def get_residues_idx_in_seq2_not_seq1(seq1: str, seq2: str):
    """
    seq1: MS-E-
    seq2: MSDER
    return: [2,4]
    """ 

    alignments = pairwise2.align.globalxx(seq1, seq2)

    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    residues_idx = [] 
    for i in range(0,len(seq1_aligned)):
        if seq1_aligned[i] == '-' and seq2_aligned[i] != '-':
            num_gaps = seq2_aligned[0:i].count('-')
            residues_idx.append(i-num_gaps)

    return residues_idx
   

def align_seqs(seq1, seq2):

    alignments = pairwise2.align.globalxx(seq1, seq2)
    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB
    logger.info('Aligned PDB1 seq with PDB2 seq:')
    logger.info(format_alignment(*alignments[0]))

    seq1_aligned_to_original_idx_mapping = {} #maps each index in pdb_seq_aligned to the corresponding index in pdb_seq (i.e accounting for gaps) 
    for i in range(0,len(seq1_aligned)):
        if seq1_aligned[i] != '-': 
            seq1_aligned_to_original_idx_mapping[i] = i-seq1_aligned[0:i].count('-')

    seq2_aligned_to_original_idx_mapping = {} #maps each index in pdb_seq_aligned to the corresponding index in pdb_seq (i.e accounting for gaps) 
    for i in range(0,len(seq2_aligned)):
        if seq2_aligned[i] != '-': 
            seq2_aligned_to_original_idx_mapping[i] = i-seq2_aligned[0:i].count('-')

    return seq1_aligned, seq2_aligned, seq1_aligned_to_original_idx_mapping, seq2_aligned_to_original_idx_mapping


def get_flanking_residues_idx(seq1: str, seq2: str):
    """
    seq1: MSDE
    seq2: -SD-
    return: seq1: [0,3], seq2: []
    """ 

    def get_nterm_gap_idx(seq_aligned):
        nterm_idx = [] 
        i = 0 
        while i < len(seq_aligned):
            if seq_aligned[i] == '-':
                nterm_idx.append(i)
                i += 1
            else:
                break  
        return nterm_idx
    def get_cterm_gap_idx(seq1_aligned, seq2_aligned):
        cterm_idx = [] 
        i = len(seq1_aligned) 
        while i > 0:
            if seq1_aligned[i-1] == '-':
                num_gaps = seq2_aligned[0:(i-1)].count('-') #need to use seq2 for number of gaps to get correct idx 
                cterm_idx.append(i-1-num_gaps)
                i -= 1
            else: 
                break
        return cterm_idx

    alignments = pairwise2.align.globalxx(seq1, seq2)

    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    flanking_residues_dict = {} 
    #if seq1 (seq2) has a gap, then the corresponding  
    #residue in seq2 (seq1) can be considered flanking  
    flanking_residues_dict['seq2_nterm'] = get_nterm_gap_idx(seq1_aligned)
    flanking_residues_dict['seq2_cterm'] = get_cterm_gap_idx(seq1_aligned, seq2_aligned)
    flanking_residues_dict['seq1_nterm'] = get_nterm_gap_idx(seq2_aligned)
    flanking_residues_dict['seq1_cterm'] = get_cterm_gap_idx(seq2_aligned, seq1_aligned)

    return flanking_residues_dict

