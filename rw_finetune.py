import argparse
import logging
import os
import random
import sys
import time
import glob
import pickle 

from torch import nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldDataModule, OpenFoldMultimerDataModule
from openfold.data import feature_pipeline
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants, protein 
from openfold.utils.argparse_utils import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)

from openfold.utils.script_utils import prep_output, get_model_basename, parse_fasta 

from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_global_step_from_zero_checkpoint
)

from pytorch_lightning.loggers import CSVLogger
from openfold.utils.logger import PerformanceLoggingCallback

import json 
import re 
from intrinsic_ft import modify_with_intrinsic_model
from collections import defaultdict

from custom_openfold_utils.pdb_utils import convert_pdb_to_mmcif, align_and_get_rmsd
from custom_openfold_utils.conformation_utils import get_residues_ignore_idx_between_af_conformations

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

SEQ_LEN_EXTRAMSA_THRESHOLD = 600 #if length of seq > 600, extra_msa is disabled so backprop doesn't crash

class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, module_config_data, hp_config_data, target_structure_path, train_alignment_dir_w_file_id, save_structure_ouptut):
        super(OpenFoldWrapper, self).__init__()
        self.model = AlphaFold(config)
        self.is_multimer = config.globals.is_multimer
        self.loss = AlphaFoldLoss(config.loss)
        self.config = config
        self.module_config_data = module_config_data
        self.hp_config_data = hp_config_data
        self.target_structure_path = target_structure_path
        self.train_alignment_dir_w_file_id = train_alignment_dir_w_file_id 
        self.save_structure_output = save_structure_ouptut
 
        self.cached_weights = None
        self.last_lr_step = -1

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, save_structure_output, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():

            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )


        if save_structure_output:
            outputs = tensor_tree_map(lambda x: np.array(x.detach().cpu().to(torch.float32)), outputs)
            outputs["final_atom_positions"] = np.squeeze(outputs["final_atom_positions"])
            outputs["final_atom_mask"] = np.squeeze(outputs["final_atom_mask"])
            outputs["plddt"] = np.squeeze(outputs["plddt"])
            
            if "weighted_ptm_score" in outputs:
                outputs["weighted_ptm_score"] = np.squeeze(outputs["weighted_ptm_score"])
                weighted_ptm_score = outputs["weighted_ptm_score"]
            else:
                weighted_ptm_score = None 

            mean_plddt = np.mean(outputs["plddt"])

            feature_processor = feature_pipeline.FeaturePipeline(self.config.data)

            pattern = "%s/features.pkl" % self.train_alignment_dir_w_file_id
            files = glob.glob(pattern, recursive=True)
            if len(files) == 1:
                features_output_path = files[0]
            print('features.pkl path: %s' % features_output_path)
            if os.path.isfile(features_output_path):
                feature_dict = np.load(features_output_path, allow_pickle=True) #this is used for all predictions, so this assumes you are predicting a single sequence 
            else:
                raise FileNotFoundError('%s/features.pkl not found' % self.train_alignment_dir_w_file_id)

            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict', is_multimer=self.is_multimer
            )
            
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()),
                processed_feature_dict)

            # Remove the recycling dimension
            batch = tensor_tree_map(lambda t: t[..., -1], batch)
            batch["residue_index"] = feature_dict["residue_index"]
            batch["aatype"] = feature_dict["aatype"] 

            multimer_ri_gap = 1
            subtract_plddt = False
            unrelaxed_protein = prep_output(
                outputs,
                processed_feature_dict,
                feature_dict,
                feature_processor,
                args.config_preset,
                multimer_ri_gap,
                subtract_plddt
            )

            unrelaxed_output_path = '%s/step_%d.pdb' % (self.config.fine_tuning_save_dir, self.global_step)
            print('SAVING UNALIGNED PDB AT %s' % unrelaxed_output_path)

            with open(unrelaxed_output_path, 'w') as fp:
                fp.write(protein.to_pdb(unrelaxed_protein))

            print('ALIGNING WRT TO %s' % self.target_structure_path)
            rmsd = align_and_get_rmsd(self.target_structure_path, unrelaxed_output_path)
            print('SAVING ALIGNED PDB AT %s' % unrelaxed_output_path)
            if weighted_ptm_score is not None:
                print('pLDDT: %.2f, RMSD: %.2f, IPTM_PTM SCORE: %.2f' % (mean_plddt,rmsd,weighted_ptm_score))
            else:
                print('pLDDT: %.2f, RMSD: %.2f' % (mean_plddt,rmsd))

            conformation_info = {} 
            conformation_info[unrelaxed_output_path] = (self.global_step, rmsd, mean_plddt, weighted_ptm_score)
            conformation_info_fname = '%s/conformation_info_step_%d.pkl' % (self.config.fine_tuning_save_dir, self.global_step)
            with open(conformation_info_fname, 'wb') as f:
                pickle.dump(conformation_info, f)

        for k,v in other_metrics.items(): 
            print('metric: %s' % k)
            print(v)
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                on_step=False, on_epoch=True, logger=True
            )

    def training_step(self, batch, batch_idx):

        if self.config.rw_finetuning.ft_method == 'SAID':
            trainable_states = {
                param_name: param_weight.cpu()
                for param_name, param_weight in self.model.state_dict().items()
                if 'intrinsic' in param_name
            }
            #saving intrinsic parameter values from GD trajectory 
            model_fname = '%s/step_%d.pt' % (self.config.fine_tuning_save_dir, self.global_step)
            torch.save(trainable_states, model_fname) 


        ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs, self.save_structure_output)

        return loss


    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            #self.model.load_state_dict(self.ema.state_dict()["params"])

        ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss and other metrics
        batch["use_clamped_fape"] = 0.

        
        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs,
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"] #ground truth 
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score
    
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-4,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
#        return torch.optim.Adam(
#            self.model.parameters(),
#            lr=learning_rate,
#            eps=eps
#        )
        # Ignored as long as a DeepSpeed optimizer is configured


        if self.config.rw_finetuning.ft_method == 'standard':
            optimizer = torch.optim.Adam(
                params = self.get_module_params(self.module_config_data["module_to_update"], self.module_config_data["layer_to_update"]), 
                lr=learning_rate,
                eps=eps
            )
        elif self.config.rw_finetuning.ft_method == 'SAID':
            optimizer = torch.optim.Adam(
                params = self.get_SAID_module_params(),
                lr=self.hp_config_data['lr'],
                eps=eps
            )
        else:
            raise ValueError("when training model, ft_method should either be standard or SAID")     

        return {
            "optimizer": optimizer,
        }

    
    def get_SAID_module_params(self):
        n = [name for name, param in self.model.named_parameters() if 'intrinsic'  in name]
        p = [param for name, param in self.model.named_parameters() if 'intrinsic'  in name]
        print('PARAMETERS TO TUNE:')
        print(n)
        return p 

    def get_module_params(self, module_to_update, layer_to_update):
        if layer_to_update != ['all']:
            n = [name for name, param in self.model.named_parameters() if (any(module in name for module in module_to_update)) and (any(layer in name for layer in layer_to_update))]
            p = [param for name, param in self.model.named_parameters() if (any(module in name for module in module_to_update)) and (any(layer in name for layer in layer_to_update))]
        else:
            n = [name for name, param in self.model.named_parameters() if any(module in name for module in module_to_update)]
            p = [param for name, param in self.model.named_parameters() if any(module in name for module in module_to_update)]

        print('PARAMETERS TO TUNE:')
        print(n)
        return p

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def load_from_jax(self, jax_path):
        model_basename = get_model_basename(jax_path)
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(self.config, self.model, jax_path, version=model_version)


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    is_low_precision = args.precision in ["bf16-mixed", "16", "bf16", "16-true", "16-mixed", "bf16-mixed"]
    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=is_low_precision
    )

    if not(args.use_templates):
        config.model.template.enabled = False
 
    msa_files = glob.glob('%s/*.a3m' % args.train_alignment_dir)
    if len(msa_files) == 0: 
        file_id = os.listdir(args.train_alignment_dir)
        if len(file_id) > 1:
            raise ValueError("should only be a single directory under %s" % args.train_alignment_dir)
        else:
            file_id = file_id[0] #e.g 1xyz_A
            file_id_wo_chain = file_id.split('_')[0]
        train_alignment_dir_w_file_id = '%s/%s' % (args.train_alignment_dir, file_id)
    else:
        file_id = (args.train_alignment_dir).split('/')[-1]
        file_id_wo_chain = file_id.split('_')[0]
        train_alignment_dir_w_file_id = args.train_alignment_dir

    pattern = "%s/*.fasta" % train_alignment_dir_w_file_id
    files = glob.glob(pattern, recursive=True)
    if len(files) == 1:
        fasta_file = files[0]
    else:
        if len(files) > 1: 
            raise FileNotFoundError("Multiple .fasta files found in train_alignment_dir -- should only be one")
        elif len(files) == 0:
            raise FileNotFoundError("No .fasta file found in train_alignment_dir")

    fasta_fname = fasta_file.split('/')[-1].split('.')[0]
    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seqs = parse_fasta(fasta_data)
    num_chains = len(seqs)

    total_seq_len = sum(len(s) for s in seqs)
    if total_seq_len > SEQ_LEN_EXTRAMSA_THRESHOLD:
        config.data.rw_finetuning.max_extra_msa = 512

    if config.globals.is_multimer:
        with open('./rw_multimer_config.json') as f:
            rw_config_data = json.load(f)
    else:
        with open('./rw_monomer_config.json') as f:
            rw_config_data = json.load(f)

    module_config_key = 'finetuning-method_%s' % config.rw_finetuning.ft_method
    module_config_data = rw_config_data[module_config_key][args.module_config]
    module_config_data['num_chains'] = num_chains

    hp_config_data = rw_config_data['hyperparameter']['train'][args.hp_config]

    if config.globals.is_multimer:
        source_str = 'source=%s' % config.rw_finetuning.model_name         
    ft_method_str = 'ft_method=%s' % config.rw_finetuning.ft_method
    conformation_parent_dir = args.train_data_dir.split('/')[-1]
    target_str = 'target=%s' % conformation_parent_dir
    
    if args.fine_tuning_save_dir is None:
        if args.target_name is None:
            log_parent_dir = './training_logs/%s' % fasta_fname
        else:
            log_parent_dir = './training_logs/%s' % args.target_name
        if config.globals.is_multimer:
            log_child_dir = '%s/%s/%s/%s/%s' % (source_str, target_str, ft_method_str, args.module_config, args.hp_config) 
        else:
            log_child_dir = '%s/%s/%s/%s/%s' % (target_str, ft_method_str, args.module_config, args.hp_config) 

        fine_tuning_save_dir = '%s/%s' % (log_parent_dir, log_child_dir)
    else:
        fine_tuning_save_dir = args.fine_tuning_save_dir
        log_parent_dir = fine_tuning_save_dir[0:fine_tuning_save_dir.rfind('/')]
        log_child_dir = fine_tuning_save_dir.split('/')[-1] 
    
    os.makedirs(fine_tuning_save_dir, exist_ok=True)

    if os.path.isdir(fine_tuning_save_dir):
        items = os.listdir(fine_tuning_save_dir)
        sub_dir = [item for item in items if os.path.isdir(os.path.join(fine_tuning_save_dir, item))]
        version_num_list = [] 
        for d in sub_dir:
            if 'version' in d:
                version_num = int(d.split('_')[1])
                version_num_list.append(version_num)
        if len(version_num_list) > 0:
            fine_tuning_save_dir = '%s/version_%d' % (fine_tuning_save_dir, max(version_num_list)+1)
        else:
            fine_tuning_save_dir = '%s/version_0' % fine_tuning_save_dir
    else:
        fine_tuning_save_dir = '%s/version_0' % fine_tuning_save_dir
 
    config.fine_tuning_save_dir = fine_tuning_save_dir

    pattern = "%s/*.pdb" % args.train_data_dir
    files = glob.glob(pattern, recursive=True)
    if len(files) == 1:
        target_structure_path = files[0]
    else:
        raise ValueError("Either no pdb files or multiple pdb files found at %s" % args.train_data_dir)

    model_module = OpenFoldWrapper(config, module_config_data, hp_config_data, target_structure_path, train_alignment_dir_w_file_id, args.save_structure_output)

    valid_trainable_module = ['input_embedder', 'recycling_embedder', 'template_angle_embedder', 'template_pair_embedder', 
                              'template_pair_stack', 'extra_msa_stack', 'evoformer', 'structure_module', 'aux_heads', 'all']

    #######

    if(args.openfold_checkpoint_path and not(args.resume_model_weights_only)):
        if(os.path.isdir(args.openfold_checkpoint_path)):  
            last_global_step = get_global_step_from_zero_checkpoint(args.openfold_checkpoint_path)
        else:
            sd = torch.load(args.openfold_checkpoint_path)
            last_global_step = int(sd['global_step'])
        model_module.resume_last_lr_step(last_global_step)
        print("Successfully loaded last lr step...")
    elif(args.openfold_checkpoint_path and args.resume_model_weights_only):
        if(os.path.isdir(args.openfold_checkpoint_path)):
            sd = get_fp32_state_dict_from_zero_checkpoint(args.openfold_checkpoint_path)
        else:
            sd = torch.load(args.openfold_checkpoint_path)
        #sd = {('model.'+k):v for k,v in sd.items()}
        import_openfold_weights_(model=model_module.model, state_dict=sd, config=config)
        print("Successfully loaded model weights...")
    elif(args.resume_from_jax_params):
        model_module.load_from_jax(args.resume_from_jax_params)
        print(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")

    #######
    
    '''for m_name, module in dict(model_module.named_modules()).items():
        print(m_name)
        print('****')
        for c_name, layer in dict(module.named_children()).items():
            print(c_name)'''

    if config.rw_finetuning.ft_method == 'SAID':
        model_module.model = modify_with_intrinsic_model(model_module.model, module_config_data, config.globals.is_multimer)

    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    if config.globals.is_multimer:
        print("Loading Multimer Data")
        data_module = OpenFoldMultimerDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )
    else:
        print("Loading Monomer Data")
        data_module = OpenFoldDataModule(
            config=config.data, 
            batch_seed=args.seed,
            **vars(args)
        )

    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    if(args.checkpoint_every_nth_epoch is not None):
        mc = ModelCheckpoint(
            every_n_epochs=args.checkpoint_every_nth_epoch,
            auto_insert_metric_name=False,
            save_top_k=-1,
        )
        callbacks.append(mc)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val/lddt_ca",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    if(args.wandb):
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wdb_logger)

    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
        )
        if(args.wandb):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = None
 
    if(args.wandb):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")


    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs = hp_config_data['max_epochs'],
        log_every_n_steps = 1,
        default_root_dir='./',
        strategy=strategy,
        callbacks=callbacks,
        logger=CSVLogger(log_parent_dir, name=log_child_dir),
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.openfold_checkpoint_path

    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_dir", type=str, default = None, 
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "--train_alignment_dir", type=str, default= None,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "--fine_tuning_save_dir", type=str, default=None,
    )
    parser.add_argument(
        "--target_name", type=str, default=None,
        help='''Name of target structure being trained against. Defaults to name of fasta file'''
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--custom_template_pdb_id", type=str, default=None, 
        help="""String of the format PDB-ID_CHAIN-ID (e.g 4ake_A). If provided,
              this structure is used as the only template."""
    )
    parser.add_argument(
        "--max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_nth_epoch", type=int, default=None,
        help="""Checkpoint every nth training epoch"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_from_jax_params", type=str, default=None,
        help="""Path to an .npz JAX parameter file with which to initialize the model"""
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--use_templates", type=bool, default=True
    )
    parser.add_argument(
        "--module_config", type=str, default='model_config_0',
        help=(
            "module_config_x where x is a number"
        )
    )

    parser.add_argument(
        "--hp_config", type=str, default='hp_config_0',
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--initial_pred_dir", type=str, default=None,
    )
    parser.add_argument(
        "--save_structure_output", action="store_true", default=False,
    )
    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser, 
        [
            "--accelerator", 
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_epoch",
            "--reload_dataloaders_every_n_epochs",
        ]
    ) 

    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if(str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    if(args.resume_from_jax_params is not None and args.openfold_checkpoint_path is not None):
        raise ValueError("Choose between loading pretrained Jax-weights and a checkpoint-path")

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)
