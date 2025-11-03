from typing import Tuple, Any, Union, List, Type
from vtgen.generator import *
import gym
from gym import spaces
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn
import torch as t
import torch.nn.functional as F  
import torchvision.transforms.functional as TF

def compute_info_nce_loss(features, target_features, temperature=0.1):
    """
    Computes the contrastive loss between the features and the target features. The features have shape (batch_size, dim)
    and the target features have shape (batch_size, dim). The contrastive loss is computed as:
    loss = -log(exp(feat * target_feat / tau) / sum(exp(feat * target_feat / tau)))

    :param features:
    :param target_features:
    :return:
    """
    # normalize the features
    q = nn.functional.normalize(features, dim=1)
    with t.no_grad():
        k = nn.functional.normalize(target_features, dim=1)

    logits = t.mm(q, k.T.detach()) / temperature
    labels = t.arange(logits.shape[0], dtype=t.long).to(q.device)
    return nn.CrossEntropyLoss()(logits, labels)

class MViTacFeatureExtractor2(BaseFeaturesExtractor):
    """
        Combined feature extractor for Dict observation spaces.
        Builds a feature extractor for each key of the space. Input from each space
        is fed through a separate submodule (CNN or MLP, depending on input shape),
        the output features are concatenated and fed through additional MLP network ("combined").

        :param observation_space:
        :param mlp_extractor_net_arch: Architecture for mlp encoding of state features before concatentation to cnn output
        :param mlp_activation_fn: Activation Func for MLP encoding layers
        :param cnn_output_dim: Number of features to output from each CNN submodule(s)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            mlp_extractor_net_arch: Union[int, List[int]] = None,
            mlp_activation_fn: Type[nn.Module] = nn.Tanh,
            cnn_output_dim: int = 64,
            cnn_base: Type[BaseFeaturesExtractor] = NatureCNN,
            mm_hyperparams=None
    ):
        super(MViTacFeatureExtractor2, self).__init__(observation_space, features_dim=1)

        cnn_extractors = {}
        cnn_momentum_extractors = {}
        flatten_extractors = {}

        self.inter_dim = mm_hyperparams['inter_dim']
        self.intra_dim = mm_hyperparams['intra_dim']

        cnn_concat_size = 0
        flatten_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                # create online encoder
                cnn_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # create momentum encoder
                cnn_momentum_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # compute the size of the concatenated features
                cnn_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                flatten_extractors[key] = nn.Flatten()
                flatten_concat_size += get_flattened_obs_dim(subspace)

        total_concat_size = cnn_concat_size + flatten_concat_size

        # default mlp arch to empty list if not specified
        if mlp_extractor_net_arch is None:
            mlp_extractor_net_arch = []

        for layer in mlp_extractor_net_arch:
            assert isinstance(layer, int), "Error: the mlp_extractor_net_arch can only include ints"

        # once vector obs is flattened can pass it through mlp
        if (mlp_extractor_net_arch != []) and (flatten_concat_size > 0):
            mlp_extractor = create_mlp(
                flatten_concat_size,
                mlp_extractor_net_arch[-1],
                mlp_extractor_net_arch[:-1],
                mlp_activation_fn
            )
            self.mlp_extractor = nn.Sequential(*mlp_extractor)
            self.mlp_extractor_momentum = nn.Sequential(*mlp_extractor)
            final_features_dim = mlp_extractor_net_arch[-1] + cnn_concat_size
        else:
            self.mlp_extractor = None
            final_features_dim = total_concat_size

        self.cnn_extractors = nn.ModuleDict(cnn_extractors)
        self.flatten_extractors = nn.ModuleDict(flatten_extractors)
        self.cnn_momentum_extractors = nn.ModuleDict(cnn_momentum_extractors)

        # Update the features dim manually
        self._features_dim = final_features_dim

        # # create heads for intra and inter modalities
        # self.observation_space_shape_visual = observation_space.spaces['visual'].shape
        # self.observation_space_shape_tactile = observation_space.spaces['tactile'].shape

        # vision heads
        self.vision_head_intra_q, self.vision_head_inter_q = self.create_heads()
        self.vision_head_intra_k, self.vision_head_inter_k = self.create_heads()

        # tactile heads
        self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_heads()
        self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_heads()

        # Initialize key encoders with query encoder weights
        self.m = 0.99  # Momentum factor for key encoder updates
        self.momentum_update_key_encoder()

        self.temperature = mm_hyperparams['temperature']
        self.weight_intra_vision = mm_hyperparams['weight_intra_vision']
        self.weight_intra_tactile = mm_hyperparams['weight_intra_tactile']
        self.weight_inter_tac_vis = mm_hyperparams['weight_inter_tac_vis']
        self.weight_inter_vis_tac = mm_hyperparams['weight_inter_vis_tac']

    def forward(self, observations: TensorDict) -> t.Tensor:
        # encode image obs through cnn
        cnn_encoded_tensor_list = []
        for key, extractor in self.cnn_extractors.items():
            x_modality = observations[key].to("cuda")
            cnn_encoded_tensor_list.append(extractor(x_modality))

        # flatten vector obs
        flatten_encoded_tensor_list = []
        for key, extractor in self.flatten_extractors.items():
            flatten_encoded_tensor_list.append(extractor(observations[key]))

        # encode combined flat vector obs through mlp extractor (if set)
        # and combine with cnn outputs
        if self.mlp_extractor is not None:
            extracted_tensor = self.mlp_extractor(t.cat(flatten_encoded_tensor_list, dim=1))
            comb_extracted_tensor = t.cat([*cnn_encoded_tensor_list, extracted_tensor], dim=1)
        else:
            comb_extracted_tensor = t.cat([*cnn_encoded_tensor_list, *flatten_encoded_tensor_list], dim=1)

        return comb_extracted_tensor

    def create_heads(self):
        head_inter = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.inter_dim)
        )

        head_intra = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.intra_dim)
        )

        return head_intra, head_inter

    def momentum_update_key_encoder(self, ) -> None:
        # Update target encoder with momentum
        for online_params, momentum_params in zip(self.cnn_extractors.parameters(),
                                                  self.cnn_momentum_extractors.parameters()):
            momentum_params.data = self.m * momentum_params.data + (1.0 - self.m) * online_params.data

    def compute_loss(self, vision_observations: t.Tensor, tactile_observations: t.Tensor) -> tuple[
        Any, Any, Any, Any, Any]:
        """
        The encode function computes the codes for the query and the key for both modalities.
        The base encoders provide the features and the projection heads provide the codes.
        :param tactile_observations:
        :param vision_observations:
        :return:
        """
        # Vision modality online encoder and heads
        vision_base_q = self.cnn_extractors['visual'](vision_observations)
        vis_queries_intra = self.vision_head_intra_q(vision_base_q)
        vis_queries_inter = self.vision_head_inter_q(vision_base_q)
        # Tactile modality online encoder and heads
        tactile_base_q = self.cnn_extractors['tactile'](tactile_observations)
        tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)
        tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)

        # Use no_grad context for the key encoders to prevent gradient updates
        with t.no_grad():
            # Vision modality momentum encoder and heads
            vision_base_k = self.cnn_momentum_extractors['visual'](vision_observations)
            vis_keys_intra = self.vision_head_intra_k(vision_base_k)
            vis_keys_inter = self.vision_head_inter_k(vision_base_k)
            # Tactile modality  momentum encoder and heads
            tactile_base_k = self.cnn_momentum_extractors['tactile'](tactile_observations)
            tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
            tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)

        # with t.no_grad():
        # Compute the contrastive loss for each pair of queries and keys
        vis_loss_intra = compute_info_nce_loss(vis_queries_intra, vis_keys_intra, self.temperature)
        tac_loss_intra = compute_info_nce_loss(tac_queries_intra, tac_keys_intra, self.temperature)
        vis_tac_inter = compute_info_nce_loss(vis_queries_inter, tac_keys_inter, self.temperature)
        tac_vis_inter = compute_info_nce_loss(tac_queries_inter, vis_keys_inter, self.temperature)

        # Combine losses
        combined_loss = (self.weight_intra_vision * vis_loss_intra
                         + self.weight_intra_tactile * tac_loss_intra
                         + self.weight_inter_tac_vis * vis_tac_inter
                         + self.weight_inter_vis_tac * tac_vis_inter)

        return combined_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scaled_dot_attn = nn.MultiheadAttention(d_model, nhead)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat1, feat2):
        # feat1 and feat2 are of shape (batch_size, seq_length, d_model)
        if len(feat1.shape) == 2:
            feat1 = feat1.unsqueeze(1)
        if len(feat2.shape) == 2:
            feat2 = feat2.unsqueeze(1)
        # Project inputs to queries, keys, and values
        queries = self.query_proj(feat1)
        keys = self.key_proj(feat2)
        values = self.value_proj(feat2)
        
        # Transpose for scaled dot-product attention
        queries = queries.permute(1, 0, 2)   # (seq_length, batch_size, d_model)
        keys = keys.permute(1, 0, 2)         # (seq_length, batch_size, d_model)
        values = values.permute(1, 0, 2)     # (seq_length, batch_size, d_model)
        
        # Apply Scaled Dot-Product Attention
        attn_output, _ = self.scaled_dot_attn(queries, keys, values)
        
        # Transpose back to original shape for output projection
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        # Project output and apply normalization
        output = self.output_proj(attn_output)
        output = self.norm(output + feat1)
        
        return output

class VisualTactileCMCL(BaseFeaturesExtractor):
    """
        Combined feature extractor for Dict observation spaces.
        Builds a feature extractor for each key of the space. Input from each space
        is fed through a separate submodule (CNN or MLP, depending on input shape),
        the output features are concatenated and fed through additional MLP network ("combined").

        :param observation_space:
        :param mlp_extractor_net_arch: Architecture for mlp encoding of state features before concatentation to cnn output
        :param mlp_activation_fn: Activation Func for MLP encoding layers
        :param cnn_output_dim: Number of features to output from each CNN submodule(s)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            mlp_extractor_net_arch: Union[int, List[int]] = None,
            mlp_activation_fn: Type[nn.Module] = nn.Tanh,
            cnn_output_dim: int = 64,
            cnn_base: Type[BaseFeaturesExtractor] = NatureCNN,
            mm_hyperparams=None
    ):
        super(VisualTactileCMCL, self).__init__(observation_space, features_dim=1)

        cnn_extractors = {}
        cnn_momentum_extractors = {}
        flatten_extractors = {}

        self.inter_dim = mm_hyperparams['inter_dim']
        self.intra_dim = mm_hyperparams['intra_dim']

        cnn_concat_size = 0
        flatten_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                # create online encoder
                cnn_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # create momentum encoder
                cnn_momentum_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # compute the size of the concatenated features
                cnn_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                flatten_extractors[key] = nn.Flatten()
                flatten_concat_size += get_flattened_obs_dim(subspace)

        total_concat_size = cnn_concat_size + flatten_concat_size

        # default mlp arch to empty list if not specified
        if mlp_extractor_net_arch is None:
            mlp_extractor_net_arch = []

        for layer in mlp_extractor_net_arch:
            assert isinstance(layer, int), "Error: the mlp_extractor_net_arch can only include ints"

        # once vector obs is flattened can pass it through mlp
        if (mlp_extractor_net_arch != []) and (flatten_concat_size > 0):
            mlp_extractor = create_mlp(
                flatten_concat_size,
                mlp_extractor_net_arch[-1],
                mlp_extractor_net_arch[:-1],
                mlp_activation_fn
            )
            self.mlp_extractor = nn.Sequential(*mlp_extractor)
            self.mlp_extractor_momentum = nn.Sequential(*mlp_extractor)
            final_features_dim = mlp_extractor_net_arch[-1] + cnn_concat_size
        else:
            self.mlp_extractor = None
            final_features_dim = total_concat_size

        self.cnn_extractors = nn.ModuleDict(cnn_extractors)
        self.flatten_extractors = nn.ModuleDict(flatten_extractors)
        self.cnn_momentum_extractors = nn.ModuleDict(cnn_momentum_extractors)

        # Update the features dim manually
        self._features_dim = final_features_dim

        self.cross_modal1 = CrossModalAttention(d_model=512, nhead=8)
        self.cross_modal2 = CrossModalAttention(d_model=512, nhead=8)

        # # create heads for intra and inter modalities
        # self.observation_space_shape_visual = observation_space.spaces['visual'].shape
        # self.observation_space_shape_tactile = observation_space.spaces['tactile'].shape

        # vision heads
        self.vision_head_intra_q, self.vision_head_inter_q = self.create_heads()
        self.vision_head_intra_k, self.vision_head_inter_k = self.create_heads()

        # tactile heads
        self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_heads()
        self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_heads()

        # Initialize key encoders with query encoder weights
        self.m = 0.99  # Momentum factor for key encoder updates
        self.momentum_update_key_encoder()

        self.temperature = mm_hyperparams['temperature']
        self.weight_intra_vision = mm_hyperparams['weight_intra_vision']
        self.weight_intra_tactile = mm_hyperparams['weight_intra_tactile']
        self.weight_inter_tac_vis = mm_hyperparams['weight_inter_tac_vis']
        self.weight_inter_vis_tac = mm_hyperparams['weight_inter_vis_tac']

    def forward(self, observations: TensorDict) -> t.Tensor:
        # encode image obs through cnn
        # torch.Size([64, 9, 128, 128])
        # torch.Size([64, 3, 128, 128])
        # torch.Size([64, 36])
        cnn_encoded_tensor_list = []
        for key, extractor in self.cnn_extractors.items():
            x_modality = observations[key].to("cuda")
            cnn_encoded_tensor_list.append(extractor(x_modality))

        # flatten vector obs
        flatten_encoded_tensor_list = []
        for key, extractor in self.flatten_extractors.items():
            flatten_encoded_tensor_list.append(extractor(observations[key]))

        # encode combined flat vector obs through mlp extractor (if set)
        # and combine with cnn outputs
        if self.mlp_extractor is not None:
            extracted_tensor = self.mlp_extractor(t.cat(flatten_encoded_tensor_list, dim=1))
            feature_tactile, feature_visual = cnn_encoded_tensor_list[0], cnn_encoded_tensor_list[1]
            comb_extracted_tensor = t.cat([t.cat([self.cross_modal1(feature_tactile, feature_visual).squeeze(1), self.cross_modal2(feature_visual, feature_tactile).squeeze(1)], dim=-1), extracted_tensor], dim=1)
        else:
            feature_tactile, feature_visual = cnn_encoded_tensor_list[0], cnn_encoded_tensor_list[1]
            comb_extracted_tensor = t.cat([self.cross_modal1(feature_tactile, feature_visual).squeeze(1), self.cross_modal2(feature_visual, feature_tactile).squeeze(1)], dim=-1)

        return comb_extracted_tensor

    def create_heads(self):
        head_inter = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.inter_dim)
        )

        head_intra = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.intra_dim)
        )

        return head_intra, head_inter

    def momentum_update_key_encoder(self, ) -> None:
        # Update target encoder with momentum
        for online_params, momentum_params in zip(self.cnn_extractors.parameters(),
                                                  self.cnn_momentum_extractors.parameters()):
            momentum_params.data = self.m * momentum_params.data + (1.0 - self.m) * online_params.data

    def compute_loss(self, vision_observations: t.Tensor, tactile_observations: t.Tensor) -> tuple[
        Any, Any, Any, Any, Any]:
        """
        The encode function computes the codes for the query and the key for both modalities.
        The base encoders provide the features and the projection heads provide the codes.
        :param tactile_observations:
        :param vision_observations:
        :return:
        """
        # Vision modality online encoder and heads
        vision_base_q = self.cnn_extractors['visual'](vision_observations)
        vis_queries_intra = self.vision_head_intra_q(vision_base_q)
        vis_queries_inter = self.vision_head_inter_q(vision_base_q)
        # Tactile modality online encoder and heads
        tactile_base_q = self.cnn_extractors['tactile'](tactile_observations)
        tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)
        tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)

        # Use no_grad context for the key encoders to prevent gradient updates
        with t.no_grad():
            # Vision modality momentum encoder and heads
            vision_base_k = self.cnn_momentum_extractors['visual'](vision_observations)
            vis_keys_intra = self.vision_head_intra_k(vision_base_k)
            vis_keys_inter = self.vision_head_inter_k(vision_base_k)
            # Tactile modality  momentum encoder and heads
            tactile_base_k = self.cnn_momentum_extractors['tactile'](tactile_observations)
            tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
            tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)

        # with t.no_grad():
        # Compute the contrastive loss for each pair of queries and keys
        vis_loss_intra = compute_info_nce_loss(vis_queries_intra, vis_keys_intra, self.temperature)
        tac_loss_intra = compute_info_nce_loss(tac_queries_intra, tac_keys_intra, self.temperature)
        vis_tac_inter = compute_info_nce_loss(vis_queries_inter, tac_keys_inter, self.temperature)
        tac_vis_inter = compute_info_nce_loss(tac_queries_inter, vis_keys_inter, self.temperature)

        # Combine losses
        combined_loss = (self.weight_intra_vision * vis_loss_intra
                         + self.weight_intra_tactile * tac_loss_intra
                         + self.weight_inter_tac_vis * vis_tac_inter
                         + self.weight_inter_vis_tac * tac_vis_inter)

        return combined_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter

class VisualCMCL_atten(BaseFeaturesExtractor):
    """
        Combined feature extractor for Dict observation spaces.
        Builds a feature extractor for each key of the space. Input from each space
        is fed through a separate submodule (CNN or MLP, depending on input shape),
        the output features are concatenated and fed through additional MLP network ("combined").

        :param observation_space:
        :param mlp_extractor_net_arch: Architecture for mlp encoding of state features before concatentation to cnn output
        :param mlp_activation_fn: Activation Func for MLP encoding layers
        :param cnn_output_dim: Number of features to output from each CNN submodule(s)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            mlp_extractor_net_arch: Union[int, List[int]] = None,
            mlp_activation_fn: Type[nn.Module] = nn.Tanh,
            cnn_output_dim: int = 64,
            cnn_base: Type[BaseFeaturesExtractor] = NatureCNN,
            mm_hyperparams=None
    ):
        super(VisualCMCL_atten, self).__init__(observation_space, features_dim=1)

        self.generator = ResnetGenerator2(  
            input_shape=(9, 128, 128),  
            output_channels=1,  
            dim=64  
        )  
        checkpoint = t.load('vtgen/ckpts/best_model.pth', map_location='cpu') # Load VT-Gen's ckpt here
        self.generator.load_state_dict(checkpoint['generator_state_dict'])  
        for param in self.generator.parameters():  
            param.requires_grad = False  
        self.generator.eval() 

        cnn_extractors = {}
        cnn_momentum_extractors = {}
        flatten_extractors = {}

        self.inter_dim = mm_hyperparams['inter_dim']
        self.intra_dim = mm_hyperparams['intra_dim']

        cnn_concat_size = 0
        flatten_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                # create online encoder
                cnn_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # create momentum encoder
                cnn_momentum_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # compute the size of the concatenated features
                cnn_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                flatten_extractors[key] = nn.Flatten()
                flatten_concat_size += get_flattened_obs_dim(subspace)
        total_concat_size = cnn_concat_size + flatten_concat_size

        # default mlp arch to empty list if not specified
        if mlp_extractor_net_arch is None:
            mlp_extractor_net_arch = []

        for layer in mlp_extractor_net_arch:
            assert isinstance(layer, int), "Error: the mlp_extractor_net_arch can only include ints"

        # once vector obs is flattened can pass it through mlp
        if (mlp_extractor_net_arch != []) and (flatten_concat_size > 0):
            mlp_extractor = create_mlp(
                flatten_concat_size,
                mlp_extractor_net_arch[-1],
                mlp_extractor_net_arch[:-1],
                mlp_activation_fn
            )
            self.mlp_extractor = nn.Sequential(*mlp_extractor)
            self.mlp_extractor_momentum = nn.Sequential(*mlp_extractor)
            final_features_dim = mlp_extractor_net_arch[-1] + cnn_concat_size
        else:
            self.mlp_extractor = None
            final_features_dim = total_concat_size

        self.cnn_extractors = nn.ModuleDict(cnn_extractors)
        self.flatten_extractors = nn.ModuleDict(flatten_extractors)
        self.cnn_momentum_extractors = nn.ModuleDict(cnn_momentum_extractors)

        # Update the features dim manually
        self._features_dim = final_features_dim

        self.cross_modal1 = CrossModalAttention(d_model=512, nhead=8)
        self.cross_modal2 = CrossModalAttention(d_model=512, nhead=8)

        # # create heads for intra and inter modalities
        # self.observation_space_shape_visual = observation_space.spaces['visual'].shape
        # self.observation_space_shape_tactile = observation_space.spaces['tactile'].shape

        # vision heads
        self.vision_head_intra_q, self.vision_head_inter_q = self.create_heads()
        self.vision_head_intra_k, self.vision_head_inter_k = self.create_heads()

        # tactile heads
        self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_heads()
        self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_heads()

        # Initialize key encoders with query encoder weights
        self.m = 0.99  # Momentum factor for key encoder updates
        self.momentum_update_key_encoder()

        self.temperature = mm_hyperparams['temperature']
        self.weight_intra_vision = mm_hyperparams['weight_intra_vision']
        self.weight_intra_tactile = mm_hyperparams['weight_intra_tactile']
        self.weight_inter_tac_vis = mm_hyperparams['weight_inter_tac_vis']
        self.weight_inter_vis_tac = mm_hyperparams['weight_inter_vis_tac']

    def forward(self, observations: TensorDict) -> t.Tensor:
        with t.no_grad():  
            observations["tactile_gt"] = observations["tactile"].cuda()
            observations['tactile'] = torch.clamp(self.generator(observations['visual'].cuda()), min=0.0)
            # observations['tactile'] = observations['tactile'].repeat(1, 3, 1, 1)
        # encode image obs through cnn
        cnn_encoded_tensor_list = []
        for key, extractor in self.cnn_extractors.items():
            x_modality = observations[key].to("cuda")
            cnn_encoded_tensor_list.append(extractor(x_modality))
        
        # flatten vector obs
        flatten_encoded_tensor_list = []
        for key, extractor in self.flatten_extractors.items():
            flatten_encoded_tensor_list.append(extractor(observations[key]))

        # encode combined flat vector obs through mlp extractor (if set)
        # and combine with cnn outputs
        if self.mlp_extractor is not None:
            extracted_tensor = self.mlp_extractor(t.cat(flatten_encoded_tensor_list, dim=1))
            feature_tactile, feature_visual = cnn_encoded_tensor_list[0], cnn_encoded_tensor_list[1]
            comb_extracted_tensor = t.cat([t.cat([self.cross_modal1(feature_tactile, feature_visual).squeeze(1), self.cross_modal2(feature_visual, feature_tactile).squeeze(1)], dim=-1), extracted_tensor], dim=1)
        else:
            feature_tactile, feature_visual = cnn_encoded_tensor_list[0], cnn_encoded_tensor_list[1]
            comb_extracted_tensor = t.cat([self.cross_modal1(feature_tactile, feature_visual).squeeze(1), self.cross_modal2(feature_visual, feature_tactile).squeeze(1)], dim=-1)

        return comb_extracted_tensor

    def create_heads(self):
        head_inter = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.inter_dim)
        )

        head_intra = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.intra_dim)
        )

        return head_intra, head_inter

    def momentum_update_key_encoder(self, ) -> None:
        # Update target encoder with momentum
        for online_params, momentum_params in zip(self.cnn_extractors.parameters(),
                                                  self.cnn_momentum_extractors.parameters()):
            momentum_params.data = self.m * momentum_params.data + (1.0 - self.m) * online_params.data

    def compute_loss(self, vision_observations: t.Tensor, tactile_observations: t.Tensor) -> tuple[
        Any, Any, Any, Any, Any]:
        """
        The encode function computes the codes for the query and the key for both modalities.
        The base encoders provide the features and the projection heads provide the codes.
        :param tactile_observations:
        :param vision_observations:
        :return:
        """
        # Vision modality online encoder and heads
        vision_base_q = self.cnn_extractors['visual'](vision_observations)
        vis_queries_intra = self.vision_head_intra_q(vision_base_q)
        vis_queries_inter = self.vision_head_inter_q(vision_base_q)
        # Tactile modality online encoder and heads
        tactile_base_q = self.cnn_extractors['tactile'](tactile_observations)
        tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)
        tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)

        # Use no_grad context for the key encoders to prevent gradient updates
        with t.no_grad():
            # Vision modality momentum encoder and heads
            vision_base_k = self.cnn_momentum_extractors['visual'](vision_observations)
            vis_keys_intra = self.vision_head_intra_k(vision_base_k)
            vis_keys_inter = self.vision_head_inter_k(vision_base_k)
            # Tactile modality  momentum encoder and heads
            tactile_base_k = self.cnn_momentum_extractors['tactile'](tactile_observations)
            tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
            tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)

        # with t.no_grad():
        # Compute the contrastive loss for each pair of queries and keys
        vis_loss_intra = compute_info_nce_loss(vis_queries_intra, vis_keys_intra, self.temperature)
        tac_loss_intra = compute_info_nce_loss(tac_queries_intra, tac_keys_intra, self.temperature)
        vis_tac_inter = compute_info_nce_loss(vis_queries_inter, tac_keys_inter, self.temperature)
        tac_vis_inter = compute_info_nce_loss(tac_queries_inter, vis_keys_inter, self.temperature)

        # Combine losses
        combined_loss = (self.weight_intra_vision * vis_loss_intra
                         + self.weight_intra_tactile * tac_loss_intra
                         + self.weight_inter_tac_vis * vis_tac_inter
                         + self.weight_inter_vis_tac * tac_vis_inter)

        return combined_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with t.no_grad():
            n_flatten = self.cnn(
                t.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: t.Tensor) -> t.Tensor:
        return self.linear(self.cnn(observations))

class VisualExtractor(BaseFeaturesExtractor):  
    """  
    Combined feature extractor for visual data and extended features.  
    
    :param observation_space: (gym.Space)  
    :param features_dim: (int) Number of features extracted.  
        This corresponds to the number of unit for the last layer.  
    """  

    def __init__(self, observation_space: gym.spaces.Dict,   
                 features_dim: int = 256,  
                 cnn_output_dim: int = 512,   
                 mlp_extractor_net_arch: list = [64, 64]):  
        super(VisualExtractor, self).__init__(observation_space, features_dim)  

        # Visual CNN extractor  
        visual_input_channels = observation_space.spaces['visual'].shape[0]  
        self.cnn_visual = nn.Sequential(  
            nn.Conv2d(visual_input_channels, 32, kernel_size=8, stride=4, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(256, cnn_output_dim, kernel_size=3, stride=1, padding=0),  
            nn.ReLU(),  
            nn.Flatten(),  
        )  

        # Compute CNN output shape  
        with t.no_grad():  
            n_flatten_visual = self.cnn_visual(  
                t.as_tensor(observation_space.spaces['visual'].sample()[None]).float()  
            ).shape[1]  

        # Extended feature MLP  
        extended_feature_dim = get_flattened_obs_dim(observation_space.spaces['extended_feature'])  
        self.extended_mlp = nn.Sequential(  
            nn.Linear(extended_feature_dim, mlp_extractor_net_arch[0]),  
            nn.ReLU(),  
            nn.Linear(mlp_extractor_net_arch[0], mlp_extractor_net_arch[1]),  
            nn.ReLU()  
        )  

        # Final fusion layers  
        combined_dim = n_flatten_visual + mlp_extractor_net_arch[-1]  
        self.fusion_layers = nn.Sequential(  
            nn.Linear(combined_dim, features_dim),  
            nn.ReLU(),  
            nn.Linear(features_dim, features_dim),  
            nn.ReLU()  
        )  

    def forward(self, observations: t.Tensor) -> t.Tensor:  
        # Process visual input  
        obs_visual = observations['visual']  
        feature_visual = self.cnn_visual(obs_visual)  

        # Process extended features  
        extended_feature = observations['extended_feature']  
        feature_extended = self.extended_mlp(extended_feature)  

        # Combine features  
        combined_features = t.cat([feature_visual, feature_extended], dim=1)  
        
        # Final processing  
        return self.fusion_layers(combined_features)
    
class TactileExtractor(BaseFeaturesExtractor):  
    """  
    :param observation_space: (gym.Space)  
    :param features_dim: (int) Number of features extracted.  
        This corresponds to the number of unit for the last layer.  
    """  

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, 
                 cnn_output_dim: int = 512, mlp_extractor_net_arch: list = [64, 64]):  
        super(TactileExtractor, self).__init__(observation_space, features_dim)  
        # We assume CxHxW images (channels first)  
        # Re-ordering will be done by pre-preprocessing or wrapper  
        tactile_input_channels = observation_space.spaces['tactile'].shape[0]  
        self.cnn_tactile = nn.Sequential(  
            nn.Conv2d(tactile_input_channels, 32, kernel_size=8, stride=4, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  # 新增层  
            nn.ReLU(),  
            nn.Conv2d(256, cnn_output_dim, kernel_size=3, stride=1, padding=0),  # 新增层  
            nn.ReLU(),  
            nn.Flatten(),  
        )  
        # Compute shape by doing one forward pass  
        with t.no_grad():  
            n_flatten_tactile = self.cnn_tactile(  
                t.as_tensor(observation_space.spaces['tactile'].sample()[None]).float()  
            ).shape[1]  
        self.linear = nn.Sequential(  
            nn.Linear(n_flatten_tactile, features_dim),  
            nn.ReLU(),  
            nn.Linear(features_dim, features_dim),  # You can adjust layers or dimensions  
            nn.ReLU()  
        )  

    def forward(self, observations: t.Tensor) -> t.Tensor:  
        obs_tactile = observations['tactile']  
        feature_tactile = self.cnn_tactile(obs_tactile)  
        ret = self.linear(feature_tactile)  
        return ret  

class VisualTactileExtractor(BaseFeaturesExtractor):  
    """  
    :param observation_space: (gym.Space)  
    :param features_dim: (int) Number of features extracted.  
        This corresponds to the number of unit for the last layer.  
    """  

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, 
                 cnn_output_dim: int = 512, mlp_extractor_net_arch: list = [64, 64]):  
        super(VisualTactileExtractor, self).__init__(observation_space, features_dim)  
        # We assume CxHxW images (channels first)  
        # Re-ordering will be done by pre-preprocessing or wrapper  
        visual_input_channels = observation_space.spaces['visual'].shape[0]  
        tactile_input_channels = observation_space.spaces['tactile'].shape[0]  
        self.cnn_visual = nn.Sequential(  
            nn.Conv2d(visual_input_channels, 32, kernel_size=8, stride=4, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  # 新增层  
            nn.ReLU(),  
            nn.Conv2d(256, cnn_output_dim, kernel_size=3, stride=1, padding=0),  # 新增层  
            nn.ReLU(),  
            nn.Flatten(),  
        )  
        self.cnn_tactile = nn.Sequential(  
            nn.Conv2d(tactile_input_channels, 32, kernel_size=8, stride=4, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  # 新增层  
            nn.ReLU(),  
            nn.Conv2d(256, cnn_output_dim, kernel_size=3, stride=1, padding=0),  # 新增层  
            nn.ReLU(),  
            nn.Flatten(),  
        )  
        # Compute shape by doing one forward pass  
        with t.no_grad():  
            n_flatten_visual = self.cnn_visual(  
                t.as_tensor(observation_space.spaces['visual'].sample()[None]).float()  
            ).shape[1]  
            n_flatten_tactile = self.cnn_tactile(  
                t.as_tensor(observation_space.spaces['tactile'].sample()[None]).float()  
            ).shape[1]  
        self.linear = nn.Sequential(  
            nn.Linear(n_flatten_visual, features_dim),  
            nn.ReLU(),  
            nn.Linear(features_dim, features_dim),  # You can adjust layers or dimensions  
            nn.ReLU()  
        )  

    def forward(self, observations: t.Tensor) -> t.Tensor:  
        obs_visual = observations['visual']  
        obs_tactile = observations['tactile']  
        feature_visual = self.cnn_visual(obs_visual)  
        feature_tactile = self.cnn_tactile(obs_tactile)  
        feature_fused = feature_visual + feature_tactile  
        ret = self.linear(feature_fused) # 1, 256  
        return ret  