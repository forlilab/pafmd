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

from run_openfold_benchmark_monomer import (
    run_msa_mask 
)

FeatureDict = MutableMapping[str, np.ndarray]

logger = logging.getLogger('wrapper')
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./ensemble_monomer.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def gen_args(alignment_dir, output_dir_base, finetuned_model_path, template_mmcif_dir, msa_mask_fraction, num_predictions_per_model, overwrite_pred):

    base_parser = argparse.ArgumentParser()

    base_parser.add_argument(
        "--benchmark_method", type=str, 
    )
    base_parser.add_argument(
        "--use_templates", type=bool
    )
    base_parser.add_argument(
        "--fasta_file", type=str, default=None,
        help="Path to FASTA file, one sequence per file. By default assumes that .fasta file is located in alignment_dir "
    )
    base_parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
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
        "--msa_mask_fraction", type=float, default=0.15
    )
    base_parser.add_argument(
        "--num_predictions_per_model", type=int, default=10
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
        "--overwrite_pred", action="store_true", default=False
    )
    base_parser.add_argument(
        "--ft_weights_dir", type=str,
        help="Directory in which AF weights are saved"
    )


    add_data_args(base_parser)
    args = base_parser.parse_args()

    openfold_checkpoint_path = finetuned_model_path
    config_preset = 'model_3_ptm'
    use_templates = False 

    args.benchmark_method = 'msa_mask'
    args.use_templates = use_templates 
    args.alignment_dir = alignment_dir
    args.output_dir_base = output_dir_base 
    args.template_mmcif_dir = template_mmcif_dir
    args.config_preset = config_preset
    args.openfold_checkpoint_path = openfold_checkpoint_path
    args.model_device = 'cuda:0'
    args.msa_mask_fraction = msa_mask_fraction 
    args.num_predictions_per_model = num_predictions_per_model
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
        "--ft_weights_dir", type=str, required=True, 
        help="Root directory in which finetuned weights are saved"
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
        "--msa_mask_fraction", type=float, default=0.15
    )
    supp_parser.add_argument(
        "--num_predictions_per_model", type=int, default=3
    )
    supp_parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )

    supp_args, _ = supp_parser.parse_known_args()


    model_pattern = '%s/**/version_0/checkpoints/epoch=49-step=50.ckpt' % supp_args.ft_weights_dir
    all_model_paths = sorted(glob.glob(model_pattern, recursive=True))

    print("TOTAL MODELS")
    print(len(all_model_paths))

    for i,finetuned_model_path in enumerate(all_model_paths):
        model_name = finetuned_model_path.split("custom_id=")[1].split("/")[0]
        print(asterisk_line)
        print("ON MODEL: %s (%d/%d)" % (model_name, i, len(all_model_paths)))
        print('RUNNING MODEL %s, located at %s' % (model_name, finetuned_model_path))
    
        args = gen_args(
            alignment_dir=supp_args.alignment_dir,
            output_dir_base='%s/model=%s' % (supp_args.output_dir_base, model_name),
            finetuned_model_path=finetuned_model_path,
            template_mmcif_dir=supp_args.template_mmcif_dir,
            msa_mask_fraction=supp_args.msa_mask_fraction,
            num_predictions_per_model=supp_args.num_predictions_per_model,
            overwrite_pred=supp_args.overwrite_pred
        )
      
        logger.info("RUNNING %s" % args.output_dir_base)
        run_msa_mask(args)
 
