# MSA Generation Overview

## Overview

Scripts to generate MSAs and corresponding input features for structure prediction via openfold for monomers or multimers. 

## Features

- Generate MSAs given either UniProt IDs or PDB IDs
- Utilize pre-computed alignments from OpenProteinSet when available
- Computes alignments from scratch when no corresponding MSAs are found in OpenProteinSet
- Create pickled feature file for downstream structure prediction

## Prerequisites

Before running the pipeline, install `custom_openfold_utils` and set up AWS credentials for OpenProteinSet access:
   
```bash
#create AWS credentials file
mkdir -p ~/.aws
vi ~/.aws/credentials
```
 
Add the following content to your credentials file:

```bash
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```


Ensure your AWS user has the proper permissions:
   - Log in to the AWS Management Console
   - Go to IAM (Identity and Access Management)
   - Attach the `AmazonS3ReadOnlyAccess` policy to your user/group. 



## Usage


### Basic Command Structure

```bash
python gen_msa_monomer.py [OPTIONS]
python gen_msa_multimer.py [OPTIONS]
```

### Common Options

| Argument | Description |
|----------|-------------|
| `--uniprot_id` | UniProt identifier for the protein of interest |
| `--pdb_id` | PDB identifier with chain (format: XXX_Y) |
| `--fasta_path` | Path to a FASTA file |
| `--msa_save_dir` | Parent directory to save the generated alignments (note: files are not directly saved in this directory, see Output section for further details) |
| `--template_mmcif_dir` | Directory containing mmCIF files for template search |

### Input Methods

The scripts can take three different input methods:

**UniProt ID**:

```bash
python ./msa_utils/gen_msa_monomer.py --uniprot_id A0A003 --msa_save_dir=./test
python ./msa_utils/gen_msa_multimer.py --uniprot_id_list A0A003,A0A003 --msa_save_dir=./test
```

 **PDB ID with chain**:

```bash
python ./msa_utils/gen_msa_monomer.py --pdb_id 6kvc_Avi --msa_save_dir=./test
python ./msa_utils/gen_msa_multimer.py --pdb_id_list 6kvc_A,6kvc_A --msa_save_dir=./test
```


**FASTA file**:

```bash
python ./msa_utils/gen_msa_monomer.py --fasta_path /path/A0A003.fasta --msa_save_dir=./test
python ./msa_utils/gen_msa_multimer.py --fasta_path /path/A0A003-A0A003.fasta --msa_save_dir=./test
```
Example FASTA files for monomer:

```bash
>XXXX_Y where XXXX corresponds to PDB ID and Y corresponds to chain num 
seq
```
Example FASTA files for multimer:

```bash
>AAAA-BBBB-CCCC_A where AAAA,BBBB,CCCC corresponds to PDB ID of each respective chain (can be the same or different)
seq1
>AAAA-BBBB-CCCC_B 
seq2
>AAAA-BBBB-CCCC_C where
seq3
```

### Database Paths

The relevant database paths also need to be passed in as inputs. We provide examples for the monomer/multimer script below:

```
python gen_msa_monomer.py --uniprot_database_path=$DOWNLOAD_DIR/uniprot/uniprot.fasta --uniref90_database_path=$DOWNLOAD_DIR/uniref90/uniref90.fasta --mgnify_database_path=$DOWNLOAD_DIR/mgnify/mgy_clusters_2022_05.fa --pdb70_database_path=$DOWNLOAD_DIR/pdb70/pdb70 --uniclust30_database_path=$DOWNLOAD_DIR/uniref30/UniRef30_2021_03 --bfd_database_path=$DOWNLOAD_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --jackhmmer_binary_path=/opt/applications/hmmer/3.3.2/gnu/bin/jackhmmer --hhblits_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhblits --hhsearch_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhsearch --kalign_binary_path=/opt/applications/kalign/2.04/gnu/bin/kalign --template_mmcif_dir=$DOWNLOAD_DIR/pdb_mmcif/mmcif_files
```


```
python gen_msa_multimer.py --uniprot_database_path=$DOWNLOAD_DIR/uniprot/uniprot.fasta --uniref90_database_path=$DOWNLOAD_DIR/uniref90/uniref90.fasta --mgnify_database_path=$DOWNLOAD_DIR/mgnify/mgy_clusters_2022_05.fa --pdb70_database_path=$DOWNLOAD_DIR/pdb70/pdb70 --uniclust30_database_path=$DOWNLOAD_DIR/uniref30/UniRef30_2021_03 --bfd_database_path=$DOWNLOAD_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --jackhmmer_binary_path=/opt/applications/hmmer/3.3.2/gnu/bin/jackhmmer --hhblits_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhblits --hhsearch_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhsearch --hmmsearch_binary_path=/opt/applications/hmmer/3.3.2/gnu/bin/hmmsearch --hmmbuild_binary_path=/opt/applications/hmmer/3.3.2/gnu/bin/hmmbuild --kalign_binary_path=/opt/applications/kalign/2.04/gnu/bin/kalign --pdb_seqres_database_path=$DOWNLOAD_DIR/pdb_seqres/pdb_seqres.txt --template_mmcif_dir=$DOWNLOAD_DIR/pdb_mmcif/mmcif_files
```
### Output 
###### File Structure
    msa_save_dir
    |
    +----> uniprot_id
           |
           +----> pdb_id
               +----> pdb_id.fasta
               +----> bfd_uniclust_hits.a3m
               +----> mgnify_hits.a3m
               +----> pdb70_hits.hhr
               +----> uniref90_hits.a3m
               +----> features.pkl
               
   
