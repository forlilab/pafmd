import copy
from functools import partial
import json
import logging
import os
import pickle
from typing import Optional, Sequence, Any, Union
import numpy as np
import glob
import sys
import re 

import ml_collections as mlc
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler
from openfold.np.residue_constants import restypes
from openfold.np.protein import from_pdb_string, to_modelcif
from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

from openfold.utils.script_utils import parse_fasta
from custom_openfold_utils.pdb_utils import get_rmsd, get_cif_string_from_pdb


class ConformationVectorFieldTrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ground_truth_data_dir: str,
                 rw_data_dir: str,
                 alignment_dir: str,
                 template_mmcif_dir: str,
                 max_template_date: str,
                 config: mlc.ConfigDict,
                 chain_data_cache_path: Optional[str] = None,
                 kalign_binary_path: str = '/usr/bin/kalign',
                 max_template_hits: int = 4,
                 obsolete_pdbs_file_path: Optional[str] = None,
                 template_release_dates_cache_path: Optional[str] = None,
                 shuffle_top_k_prefiltered: Optional[int] = None,
                 treat_pdb_as_distillation: bool = True,
                 filter_path: Optional[str] = None,
                 mode: str = "cvf_train",
                 alignment_index: Optional[Any] = None,
                 _structure_index: Optional[Any] = None,
                 ):
        """
            Args:
                ground_truth_data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                rw_data_dir:
                    A path to a directory containing PDB files derived from rw procedure
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(ConformationVectorFieldTrainDataset, self).__init__()
        self.ground_truth_data_dir = ground_truth_data_dir
        self.rw_data_dir = rw_data_dir 
        self.alignment_dir = alignment_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self.alignment_index = alignment_index
        self._structure_index = _structure_index
        self.num_conformations_to_sample = self.config.data_module.data_loaders.num_conformations_to_sample

        self.supported_exts = [".cif", ".core", ".pdb"]

        valid_modes = ["cvf_train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        if template_release_dates_cache_path is None:
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        seq_embeddings_path = os.path.join(self.ground_truth_data_dir, 'seq_embeddings_data/seq_embeddings_dict.pkl')
        conformation_vectorfield_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/conformation_vectorfield_dict.pkl')
        residues_mask_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/af_conformations_residues_mask_dict.pkl')
        uniprot_id_dict_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/uniprot_id_dict.pkl')
        template_id_rw_conformation_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/template_id_rw_conformation_path_dict.pkl')

        with open(seq_embeddings_path, 'rb') as f:
            self._seq_embeddings_dict = pickle.load(f)        
        with open(conformation_vectorfield_path, 'rb') as f:
            self._conformation_vectorfield_dict = pickle.load(f) 
        with open(residues_mask_path, 'rb') as f:
            self._residues_mask_dict = pickle.load(f) 
        with open(uniprot_id_dict_path, 'rb') as f:
            self._uniprot_id_dict = pickle.load(f) 
        with open(template_id_rw_conformation_path, 'rb') as f:
            self._template_id_rw_conformation_path_dict = pickle.load(f) 

        '''for i,key in enumerate(list(self._template_id_rw_conformation_path_dict)):
            if i != 0: 
                del self._template_id_rw_conformation_path_dict[key]''' 

        self._unique_ids = list(self._template_id_rw_conformation_path_dict.keys()) #each unique_id corresponds to uniprot_id-template_pdb_id 

        # If it's running template search for a monomer, then use hhsearch
        # as demonstrated in AlphaFold's run_alphafold.py code
        # https://github.com/deepmind/alphafold/blob/6c4d833fbd1c6b8e7c9a21dae5d4ada2ce777e10/run_alphafold.py#L462C1-L477
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_template_hits,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_cache_path,
            obsolete_pdbs_path=obsolete_pdbs_file_path,
            _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )


    def __getitem__(self, idx):
        
        curr_unique_id = self._unique_ids[idx]
        seq = self._seq_embeddings_dict[curr_unique_id][0]
        seq_embedding = {"s": self._seq_embeddings_dict[curr_unique_id][1]}
        feature_dict_all = [] #for each unique uniprot_id+template_pdb_id, all perturbed conformations are treated as a single batch  

        tensor_feature_names = self.config.common.unsupervised_features
        tensor_feature_names += self.config.common.template_features
        tensor_feature_names += ['template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask']

        all_rw_conformation_paths = self._template_id_rw_conformation_path_dict[curr_unique_id]

        if len(seq) >= 500:
            num_conformations_to_sample = self.num_conformations_to_sample//2
        else:
            num_conformations_to_sample = self.num_conformations_to_sample
    
        rw_conformation_paths_subset = np.random.choice(all_rw_conformation_paths, size=min(num_conformations_to_sample, len(all_rw_conformation_paths)), replace=False)
 
        for idx,rw_conformation_path in enumerate(rw_conformation_paths_subset):

            rw_conformation_fname = rw_conformation_path.split('/')[-1].split('.')[0]
            uniprot_id = self._uniprot_id_dict[rw_conformation_path]
            match = re.search(r'template=(\w+)', rw_conformation_path)
            template_pdb_id = match.group(1) #this is used as the template and the MSA is derived from this PDB_ID
            alignment_dir = '%s/%s/%s' % (self.alignment_dir, uniprot_id, template_pdb_id)

            print('rw conformation (%d/%d): %s' % (idx+1,len(rw_conformation_paths_subset),rw_conformation_path))

            rw_conformation_cif_string = get_cif_string_from_pdb(rw_conformation_path) 

            rw_conformation_name = rw_conformation_path.split('/')[-1].split('.')[0]
            rw_conformation_name = rw_conformation_name.replace('_unrelaxed','')
            rw_conformation_name = rw_conformation_name.replace('_relaxed','')
            rw_conformation_parent_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]

            residues_mask = self._residues_mask_dict[rw_conformation_path]
            conformation_vectorfield_spherical_coords, nearest_aligned_gtc_path, nearest_pdb_model_name = self._conformation_vectorfield_dict[rw_conformation_path]

            conformation_feats = self.data_pipeline.process_conformation(
                cif_string=rw_conformation_cif_string,
                file_id=rw_conformation_fname,
                tensor_feature_names=tensor_feature_names
            )
 
            seq_features = data_pipeline.make_sequence_features(seq, template_pdb_id, len(seq)) 
            to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
            seq_features = {
                k: to_tensor(v).squeeze(0) for k, v in seq_features.items() if k in tensor_feature_names 
            }

            phi = conformation_vectorfield_spherical_coords[1]
            theta = conformation_vectorfield_spherical_coords[2]

            phi_norm = np.stack((np.cos(phi), np.sin(phi)), axis=-1) # [N,2]
            theta_norm = np.stack((np.cos(theta), np.sin(theta)), axis=-1) # [N,2]
            normalized_phi_theta = np.stack((phi_norm,theta_norm), axis=-1) # [N,2,2]
            raw_phi_theta = np.stack((phi,theta), axis=-1) # [N,2]

            vectorfield_feats = {} 
            vectorfield_feats['normalized_phi_theta_gt'] = torch.from_numpy(normalized_phi_theta).to(torch.float32)
            vectorfield_feats['raw_phi_theta_gt'] = torch.from_numpy(raw_phi_theta).to(torch.float32)
            vectorfield_feats['residues_mask'] = torch.tensor(residues_mask, dtype=torch.int)
          
            feature_dict = {**vectorfield_feats, **conformation_feats, **seq_features, **seq_embedding}
            feature_dict_all.append(feature_dict)

        return feature_dict_all

    def __len__(self):
        return len(self._unique_ids)



class ConformationVectorFieldValidationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ground_truth_data_dir: str,
                 rw_data_dir: str,
                 alignment_dir: str,
                 template_mmcif_dir: str,
                 max_template_date: str,
                 config: mlc.ConfigDict,
                 chain_data_cache_path: Optional[str] = None,
                 kalign_binary_path: str = '/usr/bin/kalign',
                 max_template_hits: int = 4,
                 obsolete_pdbs_file_path: Optional[str] = None,
                 template_release_dates_cache_path: Optional[str] = None,
                 shuffle_top_k_prefiltered: Optional[int] = None,
                 treat_pdb_as_distillation: bool = True,
                 filter_path: Optional[str] = None,
                 mode: str = "eval",
                 alignment_index: Optional[Any] = None,
                 _structure_index: Optional[Any] = None,
                 ):
        """
            Args:
                ground_truth_data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                rw_data_dir:
                    A path to a directory containing PDB files derived from rw procedure
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(ConformationVectorFieldValidationDataset, self).__init__()
        self.ground_truth_data_dir = ground_truth_data_dir
        self.rw_data_dir = rw_data_dir 
        self.alignment_dir = alignment_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self.alignment_index = alignment_index
        self._structure_index = _structure_index
        self.num_conformations_to_sample = self.config.data_module.data_loaders.num_conformations_to_sample

        self.supported_exts = [".cif", ".core", ".pdb"]

        valid_modes = ["cvf_train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        if template_release_dates_cache_path is None:
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        seq_embeddings_path = os.path.join(self.ground_truth_data_dir, 'seq_embeddings_data/seq_embeddings_dict.pkl')
        conformation_vectorfield_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/conformation_vectorfield_dict.pkl')
        residues_mask_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/af_conformations_residues_mask_dict.pkl')
        uniprot_id_dict_path = os.path.join(self.ground_truth_data_dir, 'conformation_vectorfield_data/uniprot_id_dict.pkl')

        with open(seq_embeddings_path, 'rb') as f:
            self._seq_embeddings_dict = pickle.load(f)        
        with open(conformation_vectorfield_path, 'rb') as f:
            self._conformation_vectorfield_dict = pickle.load(f) 
        with open(residues_mask_path, 'rb') as f:
            self._residues_mask_dict = pickle.load(f) 
        with open(uniprot_id_dict_path, 'rb') as f:
            self._uniprot_id_dict = pickle.load(f) 

        self._rw_conformation_paths_all = list(self._conformation_vectorfield_dict.keys())

        # If it's running template search for a monomer, then use hhsearch
        # as demonstrated in AlphaFold's run_alphafold.py code
        # https://github.com/deepmind/alphafold/blob/6c4d833fbd1c6b8e7c9a21dae5d4ada2ce777e10/run_alphafold.py#L462C1-L477
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_template_hits,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_cache_path,
            obsolete_pdbs_path=obsolete_pdbs_file_path,
            _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )


    def __getitem__(self, idx):
        
        rw_conformation_path = self._rw_conformation_paths_all[idx]
        uniprot_id = self._uniprot_id_dict[rw_conformation_path]

        print(rw_conformation_path)

        seq = self._seq_embeddings_dict[uniprot_id][0]
        seq_embedding = {"s": self._seq_embeddings_dict[uniprot_id][1]}
        feature_dict_all = [] #for each unique uniprot_id+template_pdb_id, all perturbed conformations are treated as a single batch  

        tensor_feature_names = self.config.common.unsupervised_features
        tensor_feature_names += self.config.common.template_features
        tensor_feature_names += ['template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask']


        rw_conformation_fname = rw_conformation_path.split('/')[-1].split('.')[0]
        alignment_dir = '%s/%s' % (self.alignment_dir, uniprot_id)

        rw_conformation_cif_string = get_cif_string_from_pdb(rw_conformation_path) 

        file_id = os.listdir(alignment_dir)
        if len(file_id) > 1:
            raise ValueError("should only be a single directory under %s" % alignment_dir)
        else:
            file_id = file_id[0] #e.g 1xyz_A

        rw_conformation_name = rw_conformation_path.split('/')[-1].split('.')[0]
        rw_conformation_name = rw_conformation_name.replace('_unrelaxed','')
        rw_conformation_name = rw_conformation_name.replace('_relaxed','')
        rw_conformation_parent_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]

        residues_mask = self._residues_mask_dict[rw_conformation_path]
        conformation_vectorfield_spherical_coords, nearest_aligned_gtc_path, nearest_pdb_model_name = self._conformation_vectorfield_dict[rw_conformation_path]

        conformation_feats = self.data_pipeline.process_conformation(
            cif_string=rw_conformation_cif_string,
            file_id=rw_conformation_fname,
            tensor_feature_names=tensor_feature_names
        )

        seq_features = data_pipeline.make_sequence_features(seq, file_id, len(seq)) 
        to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
        seq_features = {
            k: to_tensor(v).squeeze(0) for k, v in seq_features.items() if k in tensor_feature_names 
        }

        phi = conformation_vectorfield_spherical_coords[1]
        theta = conformation_vectorfield_spherical_coords[2]

        phi_norm = np.stack((np.cos(phi), np.sin(phi)), axis=-1) # [N,2]
        theta_norm = np.stack((np.cos(theta), np.sin(theta)), axis=-1) # [N,2]
        normalized_phi_theta = np.stack((phi_norm,theta_norm), axis=-1) # [N,2,2]
        raw_phi_theta = np.stack((phi,theta), axis=-1) # [N,2]

        vectorfield_feats = {} 
        vectorfield_feats['normalized_phi_theta_gt'] = torch.from_numpy(normalized_phi_theta).to(torch.float32)
        vectorfield_feats['raw_phi_theta_gt'] = torch.from_numpy(raw_phi_theta).to(torch.float32)
        vectorfield_feats['residues_mask'] = torch.tensor(residues_mask, dtype=torch.int)
      
        feature_dict = {**vectorfield_feats, **conformation_feats, **seq_features, **seq_embedding}
        feature_dict_all.append(feature_dict)

        return feature_dict_all

    def __len__(self):
        return len(self._rw_conformation_paths_all)





class ConformationVectorFieldDataModule(pl.LightningDataModule):
    def __init__(self,
                 config: mlc.ConfigDict,
                 template_mmcif_dir: str,
                 max_template_date: str,
                 train_ground_truth_data_dir: Optional[str] = None,
                 train_rw_data_dir: Optional[str] = None,
                 train_alignment_dir: Optional[str] = None,
                 train_chain_data_cache_path: Optional[str] = None,
                 distillation_data_dir: Optional[str] = None,
                 distillation_alignment_dir: Optional[str] = None,
                 distillation_chain_data_cache_path: Optional[str] = None,
                 val_ground_truth_data_dir: Optional[str] = None,
                 val_rw_data_dir: Optional[str] = None,
                 val_alignment_dir: Optional[str] = None,
                 predict_data_dir: Optional[str] = None,
                 predict_alignment_dir: Optional[str] = None,
                 kalign_binary_path: str = '/usr/bin/kalign',
                 train_filter_path: Optional[str] = None,
                 distillation_filter_path: Optional[str] = None,
                 obsolete_pdbs_file_path: Optional[str] = None,
                 template_release_dates_cache_path: Optional[str] = None,
                 batch_seed: Optional[int] = None,
                 train_epoch_len: int = 50000,
                 _distillation_structure_index_path: Optional[str] = None,
                 alignment_index_path: Optional[str] = None,
                 distillation_alignment_index_path: Optional[str] = None,
                 **kwargs
                 ):
        super(ConformationVectorFieldDataModule, self).__init__()

        self.config = config
        self.template_mmcif_dir = template_mmcif_dir
        self.max_template_date = max_template_date
        self.train_ground_truth_data_dir = train_ground_truth_data_dir
        self.train_rw_data_dir = train_rw_data_dir
        self.train_alignment_dir = train_alignment_dir
        self.train_chain_data_cache_path = train_chain_data_cache_path
        self.distillation_data_dir = distillation_data_dir
        self.distillation_alignment_dir = distillation_alignment_dir
        self.distillation_chain_data_cache_path = (
            distillation_chain_data_cache_path
        )
        self.val_ground_truth_data_dir = val_ground_truth_data_dir
        self.val_rw_data_dir = val_rw_data_dir
        self.val_alignment_dir = val_alignment_dir
        self.kalign_binary_path = kalign_binary_path
        self.train_filter_path = train_filter_path
        self.distillation_filter_path = distillation_filter_path
        self.template_release_dates_cache_path = (
            template_release_dates_cache_path
        )
        self.obsolete_pdbs_file_path = obsolete_pdbs_file_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        self.config_preset = kwargs['config_preset']

        if self.train_ground_truth_data_dir is None and self.val_ground_truth_data_dir is None:
            raise ValueError(
                'At least one of train_ground_truth_data_dir or val_ground_truth_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_ground_truth_data_dir is not None

        if self.training_mode and train_alignment_dir is None:
            raise ValueError(
                'In training mode, train_alignment_dir must be specified'
            )
        elif self.val_ground_truth_data_dir is not None and val_alignment_dir is None:
            raise ValueError(
                'If val_ground_truth_data_dir is specified, val_alignment_dir must '
                'be specified as well'
            )

        # An ad-hoc measure for our particular filesystem restrictions
        self._distillation_structure_index = None
        if _distillation_structure_index_path is not None:
            with open(_distillation_structure_index_path, "r") as fp:
                self._distillation_structure_index = json.load(fp)

        self.alignment_index = None
        if alignment_index_path is not None:
            with open(alignment_index_path, "r") as fp:
                self.alignment_index = json.load(fp)

        self.distillation_alignment_index = None
        if distillation_alignment_index_path is not None:
            with open(distillation_alignment_index_path, "r") as fp:
                self.distillation_alignment_index = json.load(fp)

    def setup(self, stage=None):

            
        dataset_gen = partial(ConformationVectorFieldTrainDataset,
                              template_mmcif_dir=self.template_mmcif_dir,
                              max_template_date=self.max_template_date,
                              config=self.config,
                              kalign_binary_path=self.kalign_binary_path,
                              template_release_dates_cache_path=self.template_release_dates_cache_path,
                              obsolete_pdbs_file_path=self.obsolete_pdbs_file_path)

        self.train_dataset = dataset_gen(
            ground_truth_data_dir=self.train_ground_truth_data_dir,
            rw_data_dir=self.train_rw_data_dir, 
            chain_data_cache_path=self.train_chain_data_cache_path,
            alignment_dir=self.train_alignment_dir,
            filter_path=self.train_filter_path,
            max_template_hits=self.config.train.max_template_hits,
            shuffle_top_k_prefiltered=self.config.train.shuffle_top_k_prefiltered,
            treat_pdb_as_distillation=False,
            mode='cvf_train',
            alignment_index=None,
        )


        dataset_gen = partial(ConformationVectorFieldValidationDataset,
                              template_mmcif_dir=self.template_mmcif_dir,
                              max_template_date=self.max_template_date,
                              config=self.config,
                              kalign_binary_path=self.kalign_binary_path,
                              template_release_dates_cache_path=self.template_release_dates_cache_path,
                              obsolete_pdbs_file_path=self.obsolete_pdbs_file_path)

        self.eval_dataset = dataset_gen(
            ground_truth_data_dir=self.val_ground_truth_data_dir,
            rw_data_dir=self.val_rw_data_dir, 
            chain_data_cache_path=self.train_chain_data_cache_path,
            alignment_dir=self.val_alignment_dir,
            filter_path=self.train_filter_path,
            max_template_hits=self.config.train.max_template_hits,
            shuffle_top_k_prefiltered=self.config.train.shuffle_top_k_prefiltered,
            treat_pdb_as_distillation=False,
            mode='eval',
            alignment_index=None,
        )



    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)
    
        if stage == "train":
            dataset = self.train_dataset
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = ConformationVectorFieldBatchCollator()

        dl = DataLoader(
            dataset,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )


        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        return self._gen_dataloader("eval")





class ConformationVectorFieldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots[0])

