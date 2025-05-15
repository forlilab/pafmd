import argparse
from argparse import Namespace
import logging
import math
import numpy as np
import pandas as pd 
import os
import shutil
import json
from collections import Counter
import re
import glob  
import sys
from datetime import date
import itertools
import time 
import ml_collections as mlc
from typing import Tuple, List, Mapping, Optional, Sequence, Any, MutableMapping, Union

from scripts.utils import add_data_args

from run_openfold_rw_monomer import (
    run_rw_pipeline
)


FeatureDict = MutableMapping[str, np.ndarray]


TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'

logger = logging.getLogger('wrapper')
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./rw_monomer.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def gen_args(alignment_dir, output_dir_base, openfold_weights_dir, af_weights_dir, template_mmcif_dir, use_templates, use_af_weights, num_rw_steps, num_training_conformations, module_config, rw_hp_config, train_hp_config, skip_bootstrap_phase, skip_gd_phase, overwrite_pred):

    base_parser = argparse.ArgumentParser()
    base_parser.add_argument(
        "--fasta_file", type=str, default=None,
        help="Path to FASTA file, one sequence per file. By default assumes that .fasta file is located in alignment_dir "
    )
    base_parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    base_parser.add_argument(
        "--custom_template_pdb_id", type=str, default=None, 
        help="""String of the format PDB-ID_CHAIN-ID (e.g 4ake_A). If provided,
              this structure is used as the only template."""
    )
    base_parser.add_argument(
        "--alignment_dir", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    base_parser.add_argument(
        "--output_dir_base", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    base_parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    base_parser.add_argument(
        "--config_preset", type=str, default="model_1",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    base_parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    base_parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    base_parser.add_argument(
        "--num_bootstrap_steps", type=int, default=50 
    )
    base_parser.add_argument(
        "--num_bootstrap_hp_tuning_steps", type=int, default=10
    )
    base_parser.add_argument(
        "--num_rw_steps", type=int, default=100
    )
    base_parser.add_argument(
        "--num_rw_hp_tuning_steps_per_round", type=int, default=10
    )
    base_parser.add_argument(
        "--num_rw_hp_tuning_rounds_total", type=int, default=2
    )
    base_parser.add_argument(
        "--early_stop_rw_hp_tuning", action="store_true", default=False,
    )
    base_parser.add_argument(
        "--num_training_conformations", type=int, default=3
    )
    base_parser.add_argument(
        "--save_training_conformations", action="store_true", default=False
    )
    base_parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    base_parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    base_parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    base_parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    base_parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    base_parser.add_argument(
        "--relax_conformation", action="store_true", default=False,
    )
    base_parser.add_argument(
        "--multimer_ri_gap", type=int, default=1,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    base_parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    base_parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    base_parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    base_parser.add_argument(
        "--cif_output", action="store_true", default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    base_parser.add_argument(
        "--use_templates", type=bool
    )
    base_parser.add_argument(
        "--msa_mask_fraction", type=float, default=0.0
    )
    base_parser.add_argument(
        "--module_config", type=str, default=None,
        help=(
            "module_config_x where x is a number"
        )
    )
    base_parser.add_argument(
        "--rw_hp_config", type=str, default=None,
        help=(
            "hp_config_x where x is a number"
        )
    )
    base_parser.add_argument(
        "--train_hp_config", type=str, default=None,
        help=(
            "train_hp_config_x wheire x is a number"
        )
    )
    base_parser.add_argument(
        "--use_local_context_manager", action="store_true", default=False,
        help=(
            """whether to use local context manager
             when generating proposals. this means 
             that the same set of intrinsic_param
             will be produced within that context
             block."""
            )
    )
    base_parser.add_argument(
        "--bootstrap_phase_only", action="store_true", default=False
    )
    base_parser.add_argument(
        "--skip_bootstrap_phase", action="store_true", default=False
    )
    base_parser.add_argument(
        "--skip_gd_phase", action="store_true", default=False
    )
    base_parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )
    base_parser.add_argument(
        "--write_summary_dir", type=bool, default=True
    )
    base_parser.add_argument(
        "--mean_plddt_threshold", type=int, default=60
    )
    base_parser.add_argument(
        "--disordered_percentage_threshold", type=int, default=80
    )
    base_parser.add_argument(
        "--log_level", type=str, default='INFO'
    )
    base_parser.add_argument(
        "--openfold_weights_dir", type=str,
        help="Directory in which openfold weights are saved"
    )
    base_parser.add_argument(
        "--af_weights_dir", type=str,
        help="Directory in which AF weights are saved"
    )


    add_data_args(base_parser)
    args = base_parser.parse_args()

    if use_templates:
        if not(use_af_weights):
            openfold_checkpoint_path = '%s/finetuning_ptm_2.pt' % openfold_weights_dir
            config_preset = 'model_1_ptm'
        else:
            jax_param_path = '%s/params_model_1_ptm.npz' % af_weights_dir
            config_preset = 'model_1_ptm'
    else:
        if not(use_af_weights):
            openfold_checkpoint_path = '%s/finetuning_no_templ_ptm_1.pt' % openfold_weights_dir
            config_preset = 'model_3_ptm'
        else:
            jax_param_path = '%s/params_model_3_ptm.npz' % af_weights_dir
            config_preset = 'model_3_ptm'

    args.use_templates = use_templates 
    args.alignment_dir = alignment_dir
    args.output_dir_base = output_dir_base 
    args.template_mmcif_dir = template_mmcif_dir
    args.config_preset = config_preset
    if not(use_af_weights):
        args.openfold_checkpoint_path = openfold_checkpoint_path
    else:
        args.jax_param_path = jax_param_path
    args.model_device = 'cuda:0'
    args.module_config = module_config
    args.rw_hp_config = rw_hp_config
    args.train_hp_config = train_hp_config
    args.num_rw_steps = num_rw_steps
    args.num_training_conformations = num_training_conformations
    args.skip_bootstrap_phase = skip_bootstrap_phase
    args.skip_gd_phase = skip_gd_phase
    args.overwrite_pred = overwrite_pred

    if(args.jax_param_path is None and args.openfold_checkpoint_path is None):
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    return args  




if __name__ == "__main__":

     
    supp_parser = argparse.ArgumentParser()

    supp_parser.add_argument(
        "--alignment_dir", type=str, required=True,
        help="Path to alignment directory"
    )
    supp_parser.add_argument(
        "--output_dir_base", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction"
    )
    supp_parser.add_argument(
        "--openfold_weights_dir", type=str,
        help="Directory in which openfold weights are saved"
    )
    supp_parser.add_argument(
        "--af_weights_dir", type=str,
        help="Directory in which AF weights are saved"
    )
    supp_parser.add_argument(
        "--template_mmcif_dir", type=str,
        help="Path to template mmcif dir"
    )
    supp_parser.add_argument(
        "--use_templates", action="store_true", default=False,
        help="Whether to use templates for structure prediction"
    )
    supp_parser.add_argument(
        "--use_af_weights", type=bool, default=True,
        help="Uses AlphaFold weights if True, otherwise uses OpenFold weights"
    )
    supp_parser.add_argument(
        "--num_rw_steps", type=int, default=100
    )
    supp_parser.add_argument(
        "--num_training_conformations", type=int, default=3
    )
    supp_parser.add_argument(
        "--module_config", type=str, default='module_config_0',
        help=(
            "module_config_x where x is a number"
        )
    )
    supp_parser.add_argument(
        "--rw_hp_config", type=str, default='hp_config_0',
        help=(
            "hp_config_x where x is a number"
        )
    )
    supp_parser.add_argument(
        "--train_hp_config", type=str, default='hp_config_1',
        help=(
            "hp_config_x wheire x is a number"
        )
    )
    supp_parser.add_argument(
        "--skip_bootstrap_phase", action="store_true", default=False
    )
    supp_parser.add_argument(
        "--skip_gd_phase", action="store_true", default=False
    )
    supp_parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )

    supp_args, _ = supp_parser.parse_known_args()
    
    args = gen_args(
        alignment_dir=supp_args.alignment_dir,
        output_dir_base=supp_args.output_dir_base,
        openfold_weights_dir=supp_args.openfold_weights_dir,
        af_weights_dir=supp_args.af_weights_dir,
        template_mmcif_dir=supp_args.template_mmcif_dir,
        use_templates=supp_args.use_templates,
        use_af_weights=supp_args.use_af_weights,
        num_rw_steps=supp_args.num_rw_steps,
        num_training_conformations=supp_args.num_training_conformations,
        module_config=supp_args.module_config,
        rw_hp_config=supp_args.rw_hp_config,
        train_hp_config=supp_args.train_hp_config,
        skip_bootstrap_phase=supp_args.skip_bootstrap_phase,
        skip_gd_phase=supp_args.skip_gd_phase,
        overwrite_pred=supp_args.overwrite_pred
    )
  
    print("RUNNING %s" % args.output_dir_base)
    run_rw_pipeline(args)
 
