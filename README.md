# Table of Contents

- [MSA Generation](#msa-generation)
- [Model Inference](#model-inference)
- [Generating Input Structures for MD](#generating-input-structures-for-md)
- [Running MD Simulations](#running-md-simulations)
- [Clustering MD Simulations into Macrostates](#clustering-md-simulations-into-macrostates)
- [References](#references)




# MSA Generation

## Overview

Scripts to generate MSAs and corresponding input features for structure prediction via openfold for monomers or multimers. 

- Generate MSAs given either UniProt IDs or PDB IDs
- Utilize pre-computed alignments from OpenProteinSet when available
- Computes alignments from scratch when no corresponding MSAs are found in OpenProteinSet
- Create pickled feature file for downstream structure prediction

## Prerequisites

Before running the pipeline, install `custom_openfold_utils` (locally via pip) and set up AWS credentials for OpenProteinSet access:
   
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

### General Arugments

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
               
               
               

# Model Inference

## Overview

Scripts to run inference via MSA subsampling (AFSample2), AF-RW (AlphaFold-RandomWalk), or AF-Ensemble. 

## Prerequisites

- [openfold](https://github.com/aqlaboratory/openfold/blob/main/environment.yml)
- [mkdssp](https://ssbio.readthedocs.io/en/latest/instructions/dssp.html)
- CUDA modules for intrinsic dimensionality (install via `python intrinsic_said_setup.py develop --user`)
- `custom_openfold_utils` (install locally via pip)


## Usage


### Basic Command Structure

```bash
AFSample2: python run_openfold_benchmark_monomer_wrapper.py [OPTIONS]
AF-RW: python run_openfold_rw_monomer_wrapper.py [OPTIONS]
AF-Ensemble: python run_openfold_ensemble_monomer_wrapper.py [OPTIONS]
```


### General Arguments

| Argument | Description |
|----------|-------------|
| `--alignment_dir` | Directory in which sequence alignment files are saved |
| `--output_dir_base` | Name of base directory in which output files are saved |
| `--template_mmcif_dir` | Directory containing mmCIF files for template search |


### Arguments for AFSample2

| Argument | Description |
|----------|-------------|
| `--openfold_weights_dir` | Directory in which OpenFold weights are saved |
| `--af_weights_dir` | Directory in which AlphaFold weights are saved |
| `--use_af_weights` | Boolean flag indicating whether to use AlphaFold weights or OpenFold weights (defaults to AlphaFold weights) |
| `--num_predictions_per_model` | Number of predictions to generate via MSA masking (defaults to 100) |


### Arguments for AF-RW

| Argument | Description |
|----------|-------------|
| `--openfold_weights_dir` | Directory in which OpenFold weights are saved |
| `--af_weights_dir` | Directory in which AlphaFold weights are saved |
| `--use_af_weights` | Boolean flag indicating whether to use AlphaFold weights or OpenFold weights (defaults to AlphaFold weights) |
| `--num_training_conformations` | Number of bootstrapped conformations used to generate parameters to run random walk (defaults to 3, increasing/decreasing this number can increase/decrease diversity of predictions with a corresponding linear increase/decrease in runtime) |
| `--num_rw_steps` | Number of predictions to generate for each set of random walk parameters (defaults to 100). The number of total predictions corresponds to `num_rw_steps`*`num_training_conformations` |
| `--module_config` | module\_config\_x where x is a number. Defaults to module\_config\_0. These configurable parameters affect which modules and layers of AlphaFold are updated. See `rw_monomer_config.json` for more info. |
| `--rw_hp_config` | hp\_config\_x where x is a number. Defaults to hp\_config\_0. These configurable parameters correspond to hyperparameters affecting the random walk. See `rw_monomer_config.json` for more info.|
| `--train_hp_config` | hp\_config\_x where x is a number. Defaults to hp\_config\_1. These configurable parameters correspond to hyperparameters used in the gradient descent phase of the pipeline. See `rw_monomer_config.json` for more info. |


### Arguments for AF-Ensemble

| Argument | Description |
|----------|-------------|
| `--ft_weights_dir` | Directory in which weights for all relevant fine-tuned models are saved |
| `--num_predictions_per_model` | Number of predictions to generate via MSA masking for each fine-tuned model (defaults to 3, increasing/decreasing this number can increase/decrease diversity of predictions with a corresponding linear increase/decrease in runtime). The total number of predcitions corresponds to `num_predictions_per_model` times the number of fine-tuned models in `ft_weights_dir` (566 by default)|


### Output 
###### AFSample2 File Structure
    output_dir_base
    |
    +----> msa_mask_fraction=15 
          |
          +---->template=none 
                |
                +---->
                    initial_pred_msa_mask_fraction-0_unrelaxed.pdb
                    pred_*_msa_mask_fraction-15_unrelaxed.pdb
                    conformation_info.pkl

###### AF-RW File Structure
    output_dir_base
    |
    +----> alternative_conformations-summary
           |
           +----> initial_pred
               +----> initial_pred_unrelaxed.pdb
           +----> rw_output
           |
           +----> target=conformation[0-9]
                +----> ACCEPTED
                    +----> *.pdb
                +----> REJECTED
                    +----> *.pdb
                +----> conformation_info.pkl
###### AF-Ensemble File Structure
    output_dir_base
    |
    +---->model=*
        |
        +----> msa_mask_fraction=15 
              |
              +---->template=none 
                    |
                    +---->
                        initial_pred_msa_mask_fraction-0_unrelaxed.pdb
                        pred_*_msa_mask_fraction-15_unrelaxed.pdb  
                        conformation_info.pkl
      
# Generating Input Structures for MD

## Overview

Clusters conformations generated by AF, extracts a representative subset, and refines each conformation to account for any structural violations.  

## Prerequisites

- `numpy`
- `pandas`
- `sklearn`
- `scipy`
- `pymol`
- `biopython`
- `pdbfixer`
- `openmm`
- `custom_openfold_utils`

## Usage


### Basic Command Structure

```bash
python gen_md_input_structures.py [OPTIONS]
```


### Arguments

| Argument | Description |
|----------|-------------|
| `--conformation_info_dir` | Parent or ancestor directory in which relevant `conformation_info.pkl` files are stored.  |
| `--initial_pred_path` | Path to prediction made by default version of AF |
| `--num_clusters` | Number of clusters to group conformations into |
| `--num_md_structures` | Number of structures to refine and use as input to MD simulations |
| `--remove_disordered_tails` | Boolean flag indicating whether or not to remove disordered tails from structures |
| `--plddt_threshold` | Only clusters conformations whose pLDDT score is greater than this threshold (which is a value between 0-100). Defaults to 70. |
| `--disordered_threshold` | Only clusters conformations whose disordered percentage is less than this threshold (which is a value between 0-100). Defaults to 70. |



### Example Usage


```
AFSample2: python gen_md_input_structures.py --conformation_info_dir=./misc_results/UniProt_ID/benchmark/msa_mask_fraction=15/template=none --initial_pred_path=./misc_results/UniProt_ID/benchmark/msa_mask_fraction=15/template=none/initial_pred_msa_mask_fraction-0_unrelaxed.pdb --num_clusters 10 --num_md_structures=50 --remove_disordered_tails 

AF-RW: python gen_md_input_structures.py --conformation_info_dir=./misc_results/UniProt_ID/rw/alternative_conformations-summary/rw_output --initial_pred_path=./misc_results/UniProt_ID/rw/alternative_conformations-summary/initial_pred/initial_pred_unrelaxed.pdb --num_clusters 10 --num_md_structures=50 --remove_disordered_tails 

AF-Ensemble: python gen_md_input_structures.py --conformation_info_dir=./misc_results/UniProt_ID/ensemble --initial_pred_path=./misc_results/UniProt_ID/benchmark/msa_mask_fraction=15/template=none/initial_pred_msa_mask_fraction-0_unrelaxed.pdb --num_clusters 10 --num_md_structures=50 --remove_disordered_tails 
```

### Output 
    conformation_info_dir
    |
    +----> md_starting_structures
          |
          +---->num_clusters=10
                |
                +---->plddt_threshold=70
                     |
                      +---->disordered_threshold=70
                           |
                             +---->openmm_refined_structures
                                +---->cluster_*_idx_*_plddt_*_openmmrefinement.pdb
                                +---->cluster_*_idx_*_plddt_*_openmmrefinement_info.json

       
# Running MD Simulations

## Overview

Runs unbiased MD simulations for structures generated in the previous part of pipeline. 


## Prerequisites

- `pdbfixer `
- `openmm `
- `parmed`
- `mdtraj`

## Usage


### Basic Command Structure

```bash
python unbiased_md_wrapper.py [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--refined_conformations_dir` | Directory pointing to the location storing the folder `openmm_refined_structures`.  |
| `--run_all_conformations` | Boolean flag specifying to run MD simulations for all conformations |
| `--run_single_conformation_per_cluster` | Boolean flag specifying to run MD simulations for only a single conformation per cluster (i.e if 10 clusters were generated, 10 simulations will run, each one starting from a specific cluster) |
| `--specific_cluster_to_run` | Only run simulations corresponding to this cluster (e.g 0, 1, 2, initial etc.) |
| `--output_dir` | Directory to save results from MD simulations |
| `--production_steps` | Number of steps to run for MD simulations (defaults to 125000000, corresponding to 250 ns) |


### Example Usage


```
#runs simulations for single conformation per cluster
python unbiased_md_wrapper.py --refined_conformations_dir=./misc_results/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=5/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures --output_dir=./misc_results/UniProt_ID/md_output/rw --run_single_conformation_per_cluster

#runs simulations for all conformations
python unbiased_md_wrapper.py --refined_conformations_dir=./misc_results/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=5/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures --output_dir=./misc_results/UniProt_ID/md_output/rw --run_all_conformations

#runs simulations for single conformation for cluster 0 only
python unbiased_md_wrapper.py --refined_conformations_dir=./misc_results/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=5/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures --output_dir=./misc_results/UniProt_ID/md_output/rw --run_single_conformation_per_cluster --specific_cluster_to_run=0

#runs simulations for single conformation for default AF prediction only (i.e initial)
python unbiased_md_wrapper.py --refined_conformations_dir=./misc_results/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=5/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures --output_dir=./misc_results/UniProt_ID/md_output/rw --run_single_conformation_per_cluster --specific_cluster_to_run=initial

#runs simulations for all conformations for cluster 0 only
python unbiased_md_wrapper.py --refined_conformations_dir=./misc_results/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=5/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures --output_dir=./misc_results/UniProt_ID/md_output/rw --run_all_conformations --specific_cluster_to_run=0

```

### Output 
    output_dir
    |
    +----> cluster*-idx*
         +---->cluster_*_idx_*_plddt_*_openmmrefinement.pdb
          +---->trajectory.nc
    +----> cluster*-idx*
         +---->cluster_*_idx_*_plddt_*_openmmrefinement.pdb
          +---->trajectory.nc

       
# Clustering MD Simulations into Macrostates


## Overview

Runs unbiased MD simulations for structures generated in the previous part of pipeline. 


## Prerequisites

- `MDAnalysis`
- `numpy`
- `pandas`
- `scipy`
- `sklearn`
- `pymol`
- `custom_openfold_utils`
- [msmhelper](https://moldyn.github.io/msmhelper/tutorials/)
- [Clustering](https://github.com/moldyn/Clustering)

## Usage


### Basic Command Structure

```bash
python msm_pipeline.py [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--uniprot_id` | UniProt identifier |
| `--traj_parent_dir` | Absolute path to parent directory in which relevant trajectory.nc files are saved |
| `--initial_pred_path` | Absolute path to where default AF prediction refined via openmm is saved |
| `--exp_pdb_ids` | If you want to retrieve experimentally determined reference pdbs to supplement your analysis, feed in a list of strings separated by commas where strings are of the format XXX\_Y, where XXXX is the pdb\_id and Y is the chain\_id|
| `--calc_similarity_md_frames_exp_pdb` | Boolean flag that if true, calculates the RMSD and TM-score between MD frames and experimentally determined reference pdbs (as specified by `exp_pdb_ids`). This adds a significant amount of extra computation and is primarily useful for detailed quantiative evaluation|
|`--TMalign_path` | Absolute path to TMalign executable. Used if `calc_similarity_md_frames_exp_pdb` is set to True |
|`--save_dir` | Absolute path to directory to save results  |
| `--skip_input_coordinate_gen` | Boolean flag that if true, skips PCA feature construction and assumes it has already been generated. |


### Example Usage


```
#without calculating similarity to experimental PDBs (faster and a more typical use case)
python msm_pipeline.py --uniprot_id=UniProt_ID --traj_parent_dir=/abs/path/to/UniProt_ID/md_output/rw --initial_pred_path=/abs/path/to/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=10/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures/initial_pred_openmm_refinement.pdb --exp_pdb_ids=XXX_Y --save_dir=/abs/path/to/misc_results/UniProt_ID/md_output/rw

#with calculating similarity to experimental PDBs (slower)
python msm_pipeline.py --uniprot_id=UniProt_ID --traj_parent_dir=/abs/path/to/UniProt_ID/md_output/rw --initial_pred_path=/abs/path/to/UniProt_ID/rw/alternative_conformations-summary/rw_output/md_starting_structures/num_clusters=10/plddt_threshold=70/disordered_threshold=70/openmm_refined_structures/initial_pred_openmm_refinement.pdb --exp_pdb_ids=XXX_Y --calc_similarity_md_frames_exp_pdb --TMalign_path=/abs/path/to/TMalign --save_dir=/abs/path/to/misc_results/UniProt_ID/md_output/rw

```

### Output 
          
    save_dir
    |
    +---->msm_pipeline_output
        |
        +---->pca_contacts (clustering results whose input features correspond PCA applied to C-alpha contacts)
              |
              +---->macrostate_info  
                     | 
                     +---->component* 
                          |
                          +---->pymol_sessions
                          |  +---->macrostate*.pse.gz
                          +---->pymol_sessions_publication (same as pymol_sessions except ribbon is shown in cylindrical representation)
                             +---->macrostate*.pse.gz
                     | 
                     +---->pymol_sessions
                        +---->cluster_summary.pse.gz 
        +---->pca_pdist (clustering results whose input features correspond PCA applied to C-alpha pairwise distances)
              |
              +---->macrostate_info  
                     | 
                     +---->component* 
                          |
                          +---->pymol_sessions
                          |  +---->macrostate*.pse.gz
                          +---->pymol_sessions_publication (same as pymol_sessions except ribbon is shown in cylindrical representation)
                             +---->macrostate*.pse.gz
                     | 
                     +---->pymol_sessions
                        +---->cluster_summary.pse.gz 
                             
                             

# References

This project was built on top of [OpenFold](https://github.com/aqlaboratory/openfold)

Citation: Ahdritz, G. et al. OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. Nat Methods 21, 1514-1524 (2024).                          
