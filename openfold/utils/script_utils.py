import json
import logging
import os
import re
import time

import numpy
import torch
from torch import nn

from openfold.model.model import AlphaFold
from openfold.np import residue_constants, protein
from openfold.np.relax import relax
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_,
)

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

from intrinsic_ft import modify_with_intrinsic_model

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def count_models_to_evaluate(openfold_checkpoint_path, jax_param_path):
    model_count = 0
    if openfold_checkpoint_path:
        model_count += len(openfold_checkpoint_path.split(","))
    if jax_param_path:
        model_count += len(jax_param_path.split(","))
    return model_count

def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]

def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def load_models_from_command_line(config, model_device, openfold_checkpoint_path, jax_param_path, output_dir):
    # Create the output directory

    multiple_model_mode = count_models_to_evaluate(openfold_checkpoint_path, jax_param_path) > 1
    if multiple_model_mode:
        logger.info(f"evaluating multiple models")

    if jax_param_path:
        for path in jax_param_path.split(","):
            model_basename = get_model_basename(path)
            model_version = "_".join(model_basename.split("_")[1:])
            model = AlphaFold(config)
            model = model.eval()
            import_jax_weights_(
                model, path, version=model_version
            )
            model = model.to(model_device)
            logger.info(
                f"Successfully loaded JAX parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, model_basename, multiple_model_mode)
            yield model, output_directory

    if openfold_checkpoint_path:
        for path in openfold_checkpoint_path.split(","):
            model = AlphaFold(config)
            model = model.eval()
            checkpoint_basename = get_model_basename(path)
            if os.path.isdir(path):
                # A DeepSpeed checkpoint
                ckpt_path = os.path.join(
                    output_dir,
                    checkpoint_basename + ".pt",
                )

                if not os.path.isfile(ckpt_path):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        path,
                        ckpt_path,
                    )
                d = torch.load(ckpt_path)
                import_openfold_weights_(model=model, state_dict=d["ema"]["params"], config=config)
            else:
                ckpt_path = path
                d = torch.load(ckpt_path)

                if "ema" in d:
                    # The public weights have had this done to them already
                    d = d["ema"]["params"]
                import_openfold_weights_(model=model, state_dict=d, config=config)

            model = model.to(model_device)
            logger.info(
                f"Loaded OpenFold parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, checkpoint_basename, multiple_model_mode)
            yield model, output_directory

    if not jax_param_path and not openfold_checkpoint_path:
        raise ValueError(
            "At least one of jax_param_path or openfold_checkpoint_path must "
            "be specified."
        )


def load_model(config, model_device, openfold_checkpoint_path, jax_param_path, enable_dropout=False):
 
    if enable_dropout:
        logger.info('Loading model with dropout enabled in evoformer module')
        config.model.structure_module.dropout_rate=0.0
        af_model = AlphaFold(config)
        af_model = af_model.eval()
        for m in af_model.modules():
            if isinstance(m,torch.nn.Dropout):
                m.train()
    else:
        af_model = AlphaFold(config)
        af_model = af_model.eval()
   
    if openfold_checkpoint_path: 
        ckpt_path = openfold_checkpoint_path
        d = torch.load(ckpt_path)
        if "ema" in d:
            # The public weights have had this done to them already
            d = d["ema"]["params"]
            import_openfold_weights_(model=af_model, state_dict=d, config=config)
        if "state_dict" in d:
            d = d["state_dict"]
            d = {k.replace('model.',''):v for k,v in d.items()}
            import_openfold_weights_(model=af_model, state_dict=d, config=config, strict=False) #strict=False to account for templates 
        logger.info(
            f"Loaded OpenFold parameters at {ckpt_path}..."
        )
    elif jax_param_path:
        ckpt_path = jax_param_path
        checkpoint_basename = get_model_basename(ckpt_path)
        model_version = "_".join(checkpoint_basename.split("_")[1:])
        import_jax_weights_(
            config, af_model, ckpt_path, version=model_version
        )
        logger.info(
            f"Successfully loaded JAX parameters at {ckpt_path}..."
        )

    af_model = af_model.to(model_device)

    return af_model


def load_model_w_intrinsic_param(config, module_config_data, model_device, openfold_checkpoint_path, jax_param_path, use_templates, intrinsic_parameter, enable_dropout=False):
 
    intrinsic_parameter = torch.tensor(intrinsic_parameter).to(model_device)
    if enable_dropout:
        logger.info('Loading model with dropout enabled in evoformer module')
        config.model.structure_module.dropout_rate=0.0
        af_model = AlphaFold(config)
        af_model = af_model.eval()
        for m in af_model.modules():
            if isinstance(m,torch.nn.Dropout):
                m.train()
    else:
        af_model = AlphaFold(config)
        af_model = af_model.eval()
   
    if openfold_checkpoint_path: 
        ckpt_path = openfold_checkpoint_path
        d = torch.load(ckpt_path)
        if "ema" in d:
            # The public weights have had this done to them already
            d = d["ema"]["params"]
        import_openfold_weights_(model=af_model, state_dict=d, config=config)
        logger.info(
            f"Loaded OpenFold parameters at {ckpt_path}..."
        )   
    elif jax_param_path:
        ckpt_path = jax_param_path
        checkpoint_basename = get_model_basename(ckpt_path)
        model_version = "_".join(checkpoint_basename.split("_")[1:])
        if use_templates:
            no_template = False
        else:
            no_template = True 
        import_jax_weights_(
            config, af_model, ckpt_path, no_template, version=model_version
        )
        logger.info(
            f"Successfully loaded JAX parameters at {ckpt_path}..."
        )

    af_model_w_intrinsic_param = modify_with_intrinsic_model(af_model, module_config_data, config.globals.is_multimer)
    af_model_w_intrinsic_param.intrinsic_parameter = nn.Parameter(intrinsic_parameter)
    af_model_w_intrinsic_param = af_model_w_intrinsic_param.to(model_device)

    return af_model_w_intrinsic_param


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file


def run_model(model, batch, tag, output_dir, return_inference_time=False):
    with torch.no_grad():
        # Temporarily disable templates if there aren't any in the batch
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any([
            "template_" in k for k in batch
        ])

        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
        #update_timings({tag: {"inference": inference_time}}, os.path.join(output_dir, "timings.json"))

        model.config.template.enabled = template_enabled

    if return_inference_time:
        return out, inference_time
    else:
        return out 


def run_model_w_intrinsic_dim(model, batch, tag, output_dir, return_inference_time=False):

    with torch.no_grad():
        # Temporarily disable templates if there aren't any in the batch
        template_enabled = model.base_model.config.template.enabled
        model.base_model.config.template.enabled = template_enabled and any([
            "template_" in k for k in batch
        ])

        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
        #update_timings({tag: {"inference": inference_time}}, os.path.join(output_dir, "timings.json"))

        model.base_model.config.template.enabled = template_enabled

    if return_inference_time:
        return out, inference_time
    else:
        return out 



def prep_output(out, batch, feature_dict, feature_processor, config_preset, multimer_ri_gap, subtract_plddt):

    if "plddt" in out:
        plddt = out["plddt"]

        plddt_b_factors = numpy.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )

        if subtract_plddt:
            plddt_b_factors = 100 - plddt_b_factors
    else:
        plddt_b_factors = None 

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if feature_processor.config.common.use_templates and "template_domain_names" in feature_dict:
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
                                :feature_processor.config.predict.max_templates
                                ]

        if "template_chain_index" in feature_dict:
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                                   :feature_processor.config.predict.max_templates
                                   ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset={config_preset}",
    ])

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - numpy.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(numpy.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def relax_protein(config, model_device, unrelaxed_protein, output_directory, output_name, cif_output=False):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # the struct_str will contain either a PDB-format or a ModelCIF format string
    struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein, cif_output=cif_output)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

    # Save the relaxed PDB.
    suffix = "_relaxed.pdb"
    if cif_output:
        suffix = "_relaxed.cif"
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}{suffix}'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(struct_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")
