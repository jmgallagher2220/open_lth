# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Product Delta Pruning Method
# By James M. Gallagher

# Purpose: The purpose of this pruning method is to investigate the impact of pruning weights that 
# have a smallest product of the difference of their current weight and their initial weight times their initial weight in a lottery ticket experiment. 
# Unlike the smallest_delta pruning method, this method keeps weights that are initially large but also change the most compared to their initial value
# so it is a hybridization of sparse_global and smallest_delta.
# 
# The amount of pruning can be adjusted with the pruning_fraction hyperparameter the same as in the sparse_global
# pruning method.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Product Delta Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of additional tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, initial_model: models.base.Model, trained_model: models.base.Model, current_mask: Mask = None):
        
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
	
        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)
        
        # Determine which layers can be pruned. This will be the same for the initial and current models
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the current model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # Get the initial model weights
        initial_weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in initial_model.state_dict().items()
                   if k in prunable_tensors}

        # Create vectors of the weights
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        initial_weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in initial_weights.items()])        
        final_weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
    
        # Calculate the product of the difference between the current weight and the initial weight
        # Multiply it by the initial weight
        for k, value in enumerate(weight_vector):
            final_weight_vector[k] = (weight_vector[k] - initial_weight_vector[k])*initial_weight_vector[k] 

        # Note the use of the abs() function to make sure absolute values are evaluated
        # The function selects number_of_weights_to_prune -1 since Python is 0 indexed. 
        # For some reason, this was not done in the sparse_global pruning method. No change made to that method for
        # experimentation purposes.
        threshold = np.sort(np.abs(final_weight_vector))[number_of_weights_to_prune - 1] # Sort is least to greatest
        
        # Keep the weights with largest value in final_weight_vector (this is not an actual weight value)
        # Note that the weights get zeroed when the final_weight_vector value is not large enough compared to the threshold
        # This will affect the math in the first component of the np.where code but this is ok since the current_mask[k] value
        # will already have been zeroed from a prior pass
        
        for k in weights:
            for index, value in enumerate(weights[k]):
                for index2, value2 in enumerate(value):
                    weights[k][index][index2] = (weights[k][index][index2] - initial_weights[k][index][index2])*initial_weights[k][index][index2]

        new_mask = Mask({k: np.where(np.abs(weights[k]) > threshold, current_mask[k], np.zeros_like(v)) for k, v in weights.items()})               
      
        for k in current_mask:
                if k not in new_mask:
                        new_mask[k] = current_mask[k]

        return new_mask
