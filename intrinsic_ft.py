#adapted from https://github.com/r-three/t-few/blob/114deced63ae722a368e06cb08ea956b20c393a6/src/models/intrinsic.py
#install cuda modules via https://github.com/r-three/t-few/blob/114deced63ae722a368e06cb08ea956b20c393a6/src/intrinsic_said_setup.py

import logging
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Set
from fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda
import contextlib 
import re

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)
 

def fast_walsh_hadamard_torched(x, axis: int = 0, normalize: bool = True):
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), "For a vector of shape %s, axis must be in [0, %d] but it is %d" % (
        orig_shape,
        len(orig_shape) - 1,
        axis,
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(torch.prod(torch.tensor(orig_shape[:axis])))]
    working_shape_post = [int(torch.prod(torch.tensor(orig_shape[axis + 1 :])))]
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / np.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


@contextlib.contextmanager
def local_np_seed(seed):
    """
    The current state of the numpy random number 
    generator is saved using get_state(). 
    The yield statement is used to define the context
    block where the code within the context manager 
    will be executed. After the code within the context 
    block is executed, the original state of the numpy 
    random number generator is restored.
    
    Effectively, this enables a the same result
    to be outputted each time code using numpy 
    rng functions is executed within this context manager. 
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def fastfood_vars(DD, seed_idx, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    with torch.random.fork_rng():
        torch.manual_seed(seed_idx)
        # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
        BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
        BB = BB * 2 - 1
        # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
        GG = torch.FloatTensor(
            LL,
        ).normal_()

    with local_np_seed(seed_idx):
        # Random permutation matrix
        Pi = torch.LongTensor(np.random.permutation(LL))

    BB.requires_grad_(False)
    GG.requires_grad_(False)
    Pi.requires_grad_(False)


    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))
    return [BB.to(device), Pi.to(device), GG.to(device), divisor.to(device), LL]


def random_vars(desired_dim, intrinsic_dim, device=0):
    """Returns a random matrix of the desired dimension."""
    R = torch.FloatTensor(desired_dim, intrinsic_dim).normal_(std=0.01).to(device)
    R.requires_grad_(False)
    divisor = torch.norm(R)
    return [R, divisor]


def fastfood_torched(x, DD: int, param_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    BB, Pi, GG, divisor, LL = param_list
    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0.0, mode="constant")
    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    dd_pad = dd_pad * BB

    # HGPi(HBX)
    mul_2 = FastWalshHadamard.apply(dd_pad)
    #mul_2 = fast_walsh_hadamard_torched(dd_pad, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_3 = mul_3 * GG

    # (HGPiHBX)
    mul_5 = FastWalshHadamard.apply(mul_3)
    #mul_5 = fast_walsh_hadamard_torched(mul_3, 0, normalize=False)


    ret = mul_5[: int(DD)]
    ret = ret / (divisor * np.sqrt(float(DD) / LL))
    return ret 




def random_torched(intrinsic_vec, param_list: Tuple[torch.Tensor, int]):
    """Random dense transform"""
    R, divisor = param_list
    result = torch.matmul(R, intrinsic_vec)
    # TODO: for now we are not normalizing with the divisor, to be added later.
    return result


class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(torch.tensor([1 / np.sqrt(float(input.size(0)))]).to(input))
        if input.is_cuda:
            return fast_walsh_hadamard_transform_cuda(input.float(), False)
        else:
            return fast_walsh_hadamard_torched(input.float(), normalize=False)

    @staticmethod
    def backward(ctx, grad_output):
        #(input,) = ctx.saved_tensors
        if grad_output.is_cuda:
            return fast_walsh_hadamard_transform_cuda(grad_output.clone().float(), False).to(grad_output)
        else:
            return fast_walsh_hadamard_torched(grad_output.clone().float(), normalize=True).to(grad_output)


class SAID_monomer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        intrinsic_dimension: int,
        module_to_update: str, 
        layer_to_update: str, 
        projection_index = str,
        said=False,
        projection="fastfood",
        device="cuda",
    ):
        super(SAID_monomer, self).__init__()

        if projection_index not in ['layer','size']:
            raise ValueError("projection_index must be either layer or size")

        self.base_model = module
        self.projection_index = projection_index        
        self.projection = projection
        self.name_base_localname = []
        self.initial_value = dict()
        self.projection_params = {}
        self.said = said
        self.device = device
        self.said_size = len(list(module.named_parameters()))

        seed_idx = 0 
        num_tunable_layers = 0
        for name, param in module.named_parameters():
            if param.requires_grad and (any(m in name for m in module_to_update)):
                if ((layer_to_update[0] == 'all') or (any(l in name for l in layer_to_update))) and ('layer_norm' not in name):
                    num_tunable_layers += 1
                    self.initial_value[name] = v0 = param.clone().detach().requires_grad_(False).to(self.device)
                    DD = np.prod(v0.size())
                    if projection_index == 'size':
                        if DD not in self.projection_params:
                            self.projection_params[DD] = self.get_projection_params(DD, seed_idx, self.device)
                    elif projection_index == 'layer':
                        self.projection_params[name] = self.get_projection_params(DD, seed_idx, self.device)
                        seed_idx += 1 
                    base, localname = module, name
                    while "." in localname:
                        prefix, localname = localname.split(".", 1)
                        base = base.__getattr__(prefix) #localname corresponds to weight/bias while base corresponds to Linear/LN 
                    self.name_base_localname.append((name, base, localname))

        #logger.info('LAYERS BEING MODIFIED:')
        #logger.info(self.projection_params.keys())
   
        self.intrinsic_dimension = intrinsic_dimension
        self.intrinsic_parameter = nn.Parameter(torch.zeros((intrinsic_dimension), device=self.device))

        if said:
            self.intrinsic_parameter_said = nn.Parameter(torch.ones((num_tunable_layers), device=self.device))

    def get_projection_params(self, DD, seed_idx, device):
        if self.projection == "fastfood":
            return fastfood_vars(DD, seed_idx, device)
        elif self.projection == "random":
            return random_vars(DD, self.intrinsic_dimension, device)

    def get_projected_param(self, intrinsic_vec, DD, projection_params, init_shape):
        if self.projection == "fastfood":
            return fastfood_torched(intrinsic_vec, DD, projection_params).view(init_shape)
            #return fastfood_torched(intrinsic_vec, DD, projection_params)
        elif self.projection == "random":
            return random_torched(intrinsic_vec, projection_params).view(init_shape)

    def forward(self, batch):
        index = 0
        logger.info('intrinsic param:')
        logger.info(self.intrinsic_parameter.data)
        for name, base, localname in self.name_base_localname:
            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)
            if self.projection_index == 'size':
                ray = self.get_projected_param(self.intrinsic_parameter, DD, self.projection_params[DD], init_shape)
            elif self.projection_index == 'layer':
                ray = self.get_projected_param(self.intrinsic_parameter, DD, self.projection_params[name], init_shape)
            if self.said:
                ray = ray * self.intrinsic_parameter_said[index]
            param = self.initial_value[name] + ray
            
            delattr(base, localname) 
            setattr(base, localname, param)
            index += 1

        module = self.base_model
        out = module(batch)
        return out 


class SAID_multimer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        intrinsic_dimension: int,
        module_to_update: str, 
        layer_to_update: str,
        projection_index: str,
        num_chains: int,  
        said=False,
        projection="fastfood",
        device="cuda",
    ):
        super(SAID_multimer, self).__init__()

        if projection_index not in ['layer','size']:
            raise ValueError("projection_index must be either layer or size")

        self.base_model = module
        self.projection_index = projection_index        
        self.projection = projection
        self.name_base_localname = []
        self.initial_value = dict()
        self.projection_params = {}
        self.scaling_index = {} 
        self.said = said
        self.device = device
        self.said_size = len(list(module.named_parameters()))
        self.num_chains = num_chains

        linear_layers_pattern = r".*linear_.*_layers.*"
        seed_idx = 0 
        num_tunable_layers = 0
        for name, param in module.named_parameters():
            if param.requires_grad and (any(m in name for m in module_to_update)):
                if ((layer_to_update[0] == 'all') or (any(l in name for l in layer_to_update))) and ('layer_norm' not in name):
                    num_tunable_layers += 1
                    self.initial_value[name] = v0 = param.clone().detach().requires_grad_(False).to(self.device)
                    DD = np.prod(v0.size())
                    if projection_index == 'size':
                        if DD not in self.projection_params:
                            self.projection_params[DD] = self.get_projection_params(DD, seed_idx, self.device)
                    elif projection_index == 'layer':
                        self.projection_params[name] = self.get_projection_params(DD, seed_idx, self.device)
                        seed_idx += 1 
                    base, localname = module, name
                    while "." in localname:
                        prefix, localname = localname.split(".", 1)
                        base = base.__getattr__(prefix) #localname corresponds to weight/bias while base corresponds to Linear/LN 
                    self.name_base_localname.append((name, base, localname))

                    if re.match(linear_layers_pattern, name):
                        layer_num = int(name.split('.')[-2])
                        self.scaling_index[name] = layer_num
                    else:
                        self.scaling_index[name] = 0 #placeholder, for model without chainmasked layers  

        if list(self.projection_params.keys()) != list(self.scaling_index.keys()):
            raise ValueError("keys in projection_params != keys in scaling_index")     
 
        #logger.info('LAYERS BEING MODIFIED:')
        #logger.info(self.projection_params.keys())
   
        self.intrinsic_dimension = intrinsic_dimension
        self.intrinsic_parameter = nn.Parameter(torch.zeros((intrinsic_dimension), device=self.device))
        self.epsilon = nn.Parameter(torch.zeros((intrinsic_dimension), requires_grad=False, device=self.device))
        self.epsilon_scaling_factor = nn.Parameter(torch.ones((num_chains), requires_grad=False, device=self.device))

        if said:
            self.intrinsic_parameter_said = nn.Parameter(torch.ones((num_tunable_layers), device=self.device))

    def get_projection_params(self, DD, seed_idx, device):
        if self.projection == "fastfood":
            return fastfood_vars(DD, seed_idx, device)
        elif self.projection == "random":
            return random_vars(DD, self.intrinsic_dimension, device)

    def get_projected_param(self, scaling_index, epsilon_scaling_factor, intrinsic_vec, epsilon, DD, projection_params, init_shape):
        if self.projection == "fastfood":
            return fastfood_torched(intrinsic_vec + epsilon*epsilon_scaling_factor[scaling_index], DD, projection_params).view(init_shape)
            #return fastfood_torched(intrinsic_vec, DD, projection_params)
        elif self.projection == "random":
            return random_torched(intrinsic_vec + epsilon*epsilon_scaling_factor[scaling_index], projection_params).view(init_shape)

    def forward(self, batch):
        index = 0
        logger.info('intrinsic param:')
        logger.info(self.intrinsic_parameter.data)
        logger.info('epsilon:')
        logger.info(self.epsilon.data)
        logger.info('epsilon scaling factor:')
        logger.info(self.epsilon_scaling_factor.data)
     
        for name, base, localname in self.name_base_localname:
            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)
            if self.projection_index == 'size':
                ray = self.get_projected_param(self.scaling_index[name], self.epsilon_scaling_factor, self.intrinsic_parameter, self.epsilon, DD, self.projection_params[DD], init_shape)
            elif self.projection_index == 'layer':
                ray = self.get_projected_param(self.scaling_index[name], self.epsilon_scaling_factor, self.intrinsic_parameter, self.epsilon, DD, self.projection_params[name], init_shape)
            if self.said:
                ray = ray * self.intrinsic_parameter_said[index]

            param = self.initial_value[name] + ray
            
            delattr(base, localname) 
            setattr(base, localname, param)
            index += 1

        module = self.base_model
        out = module(batch)
        return out         


def modify_with_intrinsic_model(model, config, is_multimer):
    if is_multimer:
        return SAID_multimer(model,config['intrinsic_dim'],config['module_to_update'],config['layer_to_update'],config['projection_index'],config['num_chains'])
    else:
        return SAID_monomer(model,config['intrinsic_dim'],config['module_to_update'],config['layer_to_update'],config['projection_index'])
