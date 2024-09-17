# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
from torcheval.metrics import PeakSignalNoiseRatio

class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device, even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()
        
        self.noise_params = torch.rand(n_losses, requires_grad=True)
    
    def forward(self, losses):
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            print("self.noise_params[i]:", self.noise_params[i])
            total_loss += (1/torch.square(self.noise_params[i]))*loss + torch.log(self.noise_params[i])
        return total_loss

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x
# Kernel for MMD Loss
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        if torch.cuda.is_available():
            self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):

        L2_distances = torch.cdist(X, X) ** 2
        print("Shape of self.get_bandwidth(L2_distances):", self.get_bandwidth(L2_distances).shape)
        print("Shape of self.bandwidth_multipliers:", self.bandwidth_multipliers.shape)
        print("Shape of L2_distances:", L2_distances.shape)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class ProbabilityPredictionNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(ProbabilityPredictionNetwork, self).__init__()
        self.probability_network = nn.Sequential(
            Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                  drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm),
            nn.Linear(embed_dim, 1),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        probabilities = self.probability_network(x)
        probabilities = torch.sigmoid(probabilities)  # Convert to probabilities in range [0, 1]
        return probabilities.squeeze(-1)

# for B, 768
def kl_divergence(p, q):
    # Ensure the division is stable using a small epsilon value
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    # Using PyTorch's batch-wise operation for KL divergence
    return (p * (p / q).log()).sum(dim=1)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class CAVMAE(nn.Module):
    """ CAV-MAE Model
    """
    def __init__(self, 
                # Basic image and audio parameters
                img_size=224, 
                audio_length=1024, 
                patch_size=16, 
                in_chans=3,

                # Embedding parameters
                embed_dim=768, 
                modality_specific_depth=11, 
                num_heads=12,

                # Decoder parameters
                decoder_embed_dim=512, 
                decoder_depth=8, 
                decoder_num_heads=16,
                mlp_ratio=4.,

                # Normalization parameters
                norm_layer=nn.LayerNorm, 
                norm_pix_loss=False,

                # Other model parameters
                tr_pos=False,
                dynamic_weighting=True, 
                model_type='vanilla',
                dynamic_weight_normalization_method='unormalized',
                knowledge_distillation=False, 
                k_value=-0.25, 
                absolute_noise=False,
                split_decoder = False,
                dual_mask = False,
                complementary = False,
                AAVV = False,
                triple_mask = False,
                kd_weight=10
                ):
        super().__init__()
        print('A CAV-MAE Model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        self.model = model_type
        self.dynamic_weight_normalization_method = dynamic_weight_normalization_method
        self.knowledge_distillation = knowledge_distillation
        self.k_value = k_value
        self.absolute_noise = absolute_noise
        self.dynamic_weighting = dynamic_weighting
        self.split_decoder = split_decoder
        self.dual_mask = dual_mask
        self.complementary = complementary
        self.AAVV = AAVV
        self.triple_mask = triple_mask
        self.kd_weight = kd_weight
        print('Use dynamic weighting: ', self.dynamic_weighting)
        print('Use knowledge distillation: ', self.knowledge_distillation)
        print('Use absolute noise: ', self.absolute_noise)
        print('Dynamic weighting normalization method: ', self.dynamic_weight_normalization_method)
        print('k value: ', self.k_value)
        print('Model: ', self.model)
        print('Split decoder: ', self.split_decoder)
        print('Dual mask: ', self.dual_mask)
        print('Complementary: ', self.complementary)
        print('AAVV: ', self.AAVV)
        print('Triple mask: ', self.triple_mask)

        # the encoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # unified branch
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(12-modality_specific_depth)])

        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)

        # the decoder part
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_audio = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_video = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_video = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        if self.split_decoder == True:
            self.decoder_blocks_a = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
            self.decoder_blocks_v = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
            self.decoder_norm_a, self.decoder_norm_v = norm_layer(decoder_embed_dim), norm_layer(decoder_embed_dim)
        else:
            self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
            self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch
        self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        self.latent_projection = nn.Linear(768, 49)
        
        # mlp with input dimension is (B, 177, 768) to output dimension (B, 177, 512) to (B, 177, 256)
        self.kld_mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        ### NEW DW ###
        n_losses = 5
        self.multi_noise_loss_module = MultiNoiseLoss(n_losses)

        self.dynamic_weight_mae = 0.5
        self.dynamic_weight_contrastive = 0.5
        
        # Random dynamic weight initialization
        # random_init = False
        # torch.manual_seed(42)  # Set the random seed
        # def init_close_values():
        #     return 0.2 * torch.rand(1, device="cuda")
        
        # To set requires grad to true or false for every parameter
        grad_log_noise_mae = True
        grad_log_noise_c = True
        grad_log_noise_latent = True
        grad_log_noise_v = True
        grad_log_noise_a = True

        # Initialize the noise with same random values within a range of 0.4 and 0.6
        # if random_init == False:
        #     random_value = init_close_values()
        #     self.log_noise_mae = torch.nn.Parameter(random_value,device="cuda", requires_grad=grad_log_noise_mae)
        #     self.log_noise_c = torch.nn.Parameter(random_value,device="cuda", requires_grad=grad_log_noise_c)
        #     self.log_noise_latent = torch.nn.Parameter(random_value,device="cuda", requires_grad=grad_log_noise_latent)
        #     self.log_noise_v = torch.nn.Parameter(random_value,device="cuda", requires_grad=grad_log_noise_v)
        #     self.log_noise_a = torch.nn.Parameter(random_value,device="cuda", requires_grad=grad_log_noise_a)
        # else:
            # With 0 initialization
            # Assuming grad_log_noise_mae, grad_log_noise_c, grad_log_noise_latent, grad_log_noise_v, grad_log_noise_a are defined somewhere in your code
        if self.dynamic_weighting == False:
            self.log_noise_mae = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            self.log_noise_c = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            self.log_noise_latent = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            if self.AAVV == True:
                self.log_noise_AA = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
                self.log_noise_VV = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            else:
                self.log_noise_AA = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
                self.log_noise_VV = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            self.log_noise_AV = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            self.log_noise_v = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            self.log_noise_a = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)    
        else:
            # self.log_noise_mae = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            # self.log_noise_c = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=True)
            # self.log_noise_latent = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            # if self.AAVV == True:
            #     self.log_noise_AA = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=True)
            #     self.log_noise_VV = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=True)
            # else:
            #     self.log_noise_AA = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            #     self.log_noise_VV = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=False)
            # self.log_noise_AV = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=True)
            # self.log_noise_v = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=True)
            # self.log_noise_a = torch.nn.Parameter(torch.full((1,), 1.0, device="cuda"), requires_grad=True)
            if self.AAVV == True:
                self.dw_fc = self.fc = nn.Linear(5, 4)
            else:
                print("USING Linear(3, 3)")
                self.dw_fc1 = self.fc = nn.Linear(3, 64)
                self.dw_fc2 = self.fc = nn.Linear(64, 128)
                self.dw_fc3 = self.fc = nn.Linear(128, 64)
                self.dw_fc4 = self.fc = nn.Linear(64, 3)

        # This comment was for the original dynamic weight formula from the paper
        """
        torch.manual_seed(42)  # Set the random seed
        def init_close_values():
            return 0.4 + 0.2 * torch.rand(1, device="cuda")
        self.noise_param_mae = torch.nn.Parameter(init_close_values(), requires_grad=True)
        self.noise_param_c = torch.nn.Parameter(init_close_values(), requires_grad=True)
        self.noise_param_latent = torch.nn.Parameter(init_close_values(), requires_grad=True)
        self.noise_param_classification = torch.nn.Parameter(init_close_values(), requires_grad=True)
        """
        # KD
        self.delta_psnr_a = 0.0
        self.delta_psnr_v = 0.0
        self.psnr_a_prev = 0.0
        self.psnr_v_prev = 0.0
        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

        self.num_patches_a = self.patch_embed_a.num_patches
        self.num_patches_v = self.patch_embed_v.num_patches

        # Probability prediction network
        self.pos_embed_probs_a = nn.Parameter(torch.zeros(1, self.num_patches_a, embed_dim))
        self.pos_embed_probs_v = nn.Parameter(torch.zeros(1, self.num_patches_v, embed_dim))
        self.probability_network = ProbabilityPredictionNetwork(embed_dim)
        self.get_token_probs = nn.Sequential(
                        Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                        drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm,
                        ),
                        nn.Linear(embed_dim, 1),
                        torch.nn.Flatten(start_dim=1),
                        )
        mask_ratio = 0.75
        self.visible_patches_a = int(self.num_patches_a*(1-mask_ratio))
        self.visible_patches_v = int(self.num_patches_v*(1-mask_ratio))
        self.softmax = nn.Softmax(dim=1)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))

        decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        torch.nn.init.normal_(self.decoder_modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, c, h, w, p=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    # def adaptive_masking(self, x, mask_ratio, epoch_id, isAudio=True, video_mask=None, audio_mask=None):
    #     """
    #     Perform adaptive masking with token shuffling based on probabilities.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim

    #     visible_patches = int(L * (1 - mask_ratio))
    #     # Calculate token probabilities
    #     if(isAudio):
    #         x_with_pos = x + self.pos_embed_probs_a.type_as(x).to(x.device).clone()
    #     else:
    #         x_with_pos = x + self.pos_embed_probs_v.type_as(x).to(x.device).clone()
    #     logits = self.get_token_probs(x_with_pos)
    #     logits = torch.nan_to_num(logits)
    #     p_x = self.softmax(logits)

    #     # Sample visible tokens based on probabilities
    #     num_to_mask = L - visible_patches
    #     mask_idx = torch.multinomial(p_x, num_samples=num_to_mask, replacement=False)

    #     # Shuffle tokens
    #     ids_shuffle = torch.randperm(L, device=x.device).repeat(N, 1)
    #     x_shuffled = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, D))

    #     # Create binary mask and apply to shuffled tokens
    #     mask = torch.ones((N, L)).to(x.device, dtype=torch.bool)
    #     mask.scatter_(dim=1, index=mask_idx, value=0)

    #     mask_token = self.mask_token.view(1, 1, -1)

    #     x_masked = x_shuffled.clone()
    #     x_masked[mask] = mask_token

    #     # Compute inverse indices for unshuffling
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     return x_masked, mask, ids_restore
    # def adaptive_masking(self, x, mask_ratio, epoch_id, isAudio=True, video_mask=None, audio_mask=None):
    #     """
    #     Perform adaptive masking based on token probabilities.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim

    #     # Calculate number of visible patches
    #     visible_patches = int(L * (1 - mask_ratio))

    #     # Add positional embeddings to x
    #     if(isAudio):
    #         x_with_pos = x + self.pos_embed_probs_a.type_as(x).to(x.device).clone()
    #     else:
    #         x_with_pos = x + self.pos_embed_probs_v.type_as(x).to(x.device).clone()
    #     logits = self.get_token_probs(x_with_pos)
    #     logits = torch.nan_to_num(logits)
    #     p_x = self.softmax(logits)

    #     # Sample visible tokens based on probabilities
    #     num_to_mask = L - visible_patches
    #     mask_idx = torch.multinomial(p_x, num_samples=num_to_mask, replacement=False)

    #     # Create binary mask
    #     mask = torch.ones((N, L), dtype=torch.bool, device=x.device)
    #     mask.scatter_(1, mask_idx, 0)

    #     # Ensure mask_token is correctly shaped
    #     if self.mask_token.shape[0] != D:
    #         raise ValueError(f"mask_token shape {self.mask_token.shape[0]} does not match feature dimension {D}")
    #     mask_token = self.mask_token.unsqueeze(0).unsqueeze(0)

    #     # Apply mask to tokens
    #     x_masked = x.clone()
    #     x_masked[mask] = mask_token

    #     return x_masked, mask

    def get_mask_a(self, x):
        N, L, D = x.shape  # batch, length, dim
        x = x + self.pos_embed_probs_a.type_as(x).to(x.device).clone() #detach()
        logits = self.get_token_probs(x)
        logits =  torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        vis_idx = torch.multinomial(p_x, num_samples=self.visible_patches_a, replacement=False)
        mask = torch.ones((x.shape[0], x.shape[1])).to(x.device, non_blocking=True)
        mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0)
        mask = mask.flatten(1).to(torch.bool)

        ids_restore = torch.arange(L, device=x.device).repeat(N, 1)
        return p_x, vis_idx, mask, ids_restore
    
    def get_mask_v(self, x):
        N, L, D = x.shape  # batch, length, dim
        x = x + self.pos_embed_probs_v.type_as(x).to(x.device).clone() #detach()
        logits = self.get_token_probs(x)
        logits =  torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        vis_idx = torch.multinomial(p_x, num_samples=self.visible_patches_v, replacement=False)

        mask = torch.ones((x.shape[0], x.shape[1])).to(x.device, non_blocking=True)
        mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0)
        mask = mask.flatten(1).to(torch.bool)

        ids_restore = torch.arange(L, device=x.device).repeat(N, 1)

        return p_x, vis_idx, mask, ids_restore 
     
    def adaptive_masking(self, x, mask_ratio, epoch_id, isAudio=True, video_mask=None, audio_mask=None):
        N, L, D = x.shape  # batch, length, dim
        visible_patches = int(L * (1 - mask_ratio))

        # Original indices of patches
        original_indices = torch.arange(L, device=x.device).expand(N, -1)

        # Assuming self.probability_network outputs the probabilities
        probabilities = self.probability_network(x)  # Shape [N, L]
        probabilities = torch.nan_to_num(probabilities)  # Handle NaNs
        p_x = F.softmax(probabilities, dim=-1)

        # Randomly select patches based on the probabilities
        vis_idx = torch.multinomial(p_x, num_samples=visible_patches, replacement=False)

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones((N, L), device=x.device)
        mask.scatter_(dim=-1, index=vis_idx, value=0.0)
        mask = mask.to(torch.bool)  # Convert to boolean mask if required

        # Apply mask to the input and original indices
        x_masked = x.masked_select(mask.unsqueeze(-1)).view(N, -1, D)  # Reshape to the correct dimensions
        ids_restore = original_indices.masked_select(mask).view(N, -1)

        return x_masked, mask, ids_restore


    def random_masking_unstructured(self, x, mask_ratio, epoch_id, isAudio=True, video_mask=None, audio_mask=None, need_two=True, need_three=False):
        # """
        # Perform per-sample random masking by per-sample shuffling.
        # Per-sample shuffling is done by argsort random noise.
        # x: [N, L, D], sequence
        # """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Use precomputed shuffling indices if provided
        if isAudio and audio_mask is not None:
            ids_shuffle = audio_mask[:, epoch_id % 4]
        elif not isAudio and video_mask is not None:
            ids_shuffle = video_mask[:, (epoch_id % 40) // 10]
        else:
            ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        if need_two:
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask.scatter_(1, ids_keep, 0)

            ids_keep_1 = ids_shuffle[:, len_keep:2*len_keep]
            x_masked_1 = torch.gather(x, dim=1, index=ids_keep_1.unsqueeze(-1).repeat(1, 1, D))

            # # Generate the binary mask: 0 is keep, 1 is remove
            mask_1 = torch.ones([N, L], device=x.device)
            mask_1.scatter_(1, ids_keep_1, 0)

            return x_masked, mask, ids_restore, x_masked_1, mask_1, ids_restore
        elif need_three:
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask.scatter_(1, ids_keep, 0)

            ids_keep_1 = ids_shuffle[:, len_keep:2*len_keep]
            x_masked_1 = torch.gather(x, dim=1, index=ids_keep_1.unsqueeze(-1).repeat(1, 1, D))

            # # Generate the binary mask: 0 is keep, 1 is remove
            mask_1 = torch.ones([N, L], device=x.device)
            mask_1.scatter_(1, ids_keep_1, 0)

            ids_keep_2 = ids_shuffle[:, 2*len_keep:3*len_keep]
            x_masked_2 = torch.gather(x, dim=1, index=ids_keep_2.unsqueeze(-1).repeat(1, 1, D))

            # # Generate the binary mask: 0 is keep, 1 is remove
            mask_2 = torch.ones([N, L], device=x.device)
            mask_2.scatter_(1, ids_keep_2, 0)

            return x_masked, mask, ids_restore, x_masked_1, mask_1, ids_restore, x_masked_2, mask_2, ids_restore
        else:
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask.scatter_(1, ids_keep, 0)

            return x_masked, mask, ids_restore

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
        if mode == 'time':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == 'freq':
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == 'tf':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, epoch_id, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured', audio_mask=None, video_mask=None):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # by default, we always use unstructured masking
        if mask_mode == 'unstructured':
            a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False)
            # print("1SHAPE IDS A", ids_restore_a.shape)
            # print(ids_restore_a)
            # p_x, vis_idx, mask_a, ids_restore_a  = self.get_mask_a(a) # adaptive mask
        # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
        else:
            p_x, vis_idx, mask_a, ids_restore_a  = self.get_mask_a(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

        # visual branch always use unstructured masking
        v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False)

        # p_x, vis_idx, mask_v, ids_restore_v  = self.get_mask_v(v)
        # a = a[~mask_a].reshape(a.shape[0], -1, a.shape[-1])  # Reshape to maintain the batch and feature dimensions
        # v = v[~mask_v].reshape(v.shape[0], -1, v.shape[-1])  # Reshape to maintain the batch and feature dimensions

        # audio and visual stream, independent blocks
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        x = torch.cat((a, v), dim=1)

        # unified stream, shared blocks_u, but independent normalization layers
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)

        for blk in self.blocks_u:
            ca = blk(a, 'a')
        ca = self.norm_a(ca)

        for blk in self.blocks_u:
            cv = blk(v, 'v')
        cv = self.norm_v(cv)
        # print("IN Encoder SHAPE_X",x.shape)
        return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv

    def forward_encoder_dual_mask(self, epoch_id, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured', audio_mask=None, video_mask=None):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # by default, we always use unstructured masking
        if mask_mode == 'unstructured':
            if self.complementary == False:
                a1, mask_a1, ids_restore_a1 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False)
                a2, mask_a2, ids_restore_a2 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False)
            else:
                a1, mask_a1, ids_restore_a1, a2, mask_a2, ids_restore_a2 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=True)

            # p_x, vis_idx, mask_a2,  ids_restore_a2 = self.get_mask_a(a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
        else:
            a1, mask_a1, ids_restore_a1 = self.random_masking_unstructured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)
            # p_x2, vis_idx2, mask_a2, ids_restore_a2 = self.get_mask_a(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

        # visual branch always use unstructured masking

        if self.complementary == False:
            v1, mask_v1, ids_restore_v1 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False)
            v2, mask_v2, ids_restore_v2 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False)
        else:
            v1, mask_v1, ids_restore_v1, v2, mask_v2, ids_restore_v2 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=True)
        # # print common elements between mask_a1 and mask_a2
        # matching_elements_count = 0
        # for m1, m2 in zip(mask_a1.flatten(), mask_a2.flatten()):
        #     if m1 == m2:
        #         matching_elements_count += 1

        # Process each masked audio and video input through their respective blocks
        for blk in self.blocks_a:
            a1 = blk(a1)
            a2 = blk(a2)  # You might need to clone the blocks if they are not stateless

        for blk in self.blocks_v:
            v1 = blk(v1)
            v2 = blk(v2)  # Similar cloning might be necessary

        x1 = torch.cat((a1, v1), dim=1)
        x2 = torch.cat((a2, v2), dim=1)

        for blk in self.blocks_u:
            x1 = blk(x1)
            x2 = blk(x2)  # Again, consider cloning if needed
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        for blk in self.blocks_u:
            ca1 = blk(a1, 'a')
            ca2 = blk(a2, 'a')
        ca1 = self.norm_a(ca1)
        ca2 = self.norm_a(ca2)

        for blk in self.blocks_u:
            cv1 = blk(v1, 'v')
            cv2 = blk(v2, 'v')
        cv1 = self.norm_v(cv1)
        cv2 = self.norm_v(cv2)

        return x1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, ca1, cv1, x2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2, ca2, cv2

    def forward_encoder_triple_mask(self, epoch_id, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured', audio_mask=None, video_mask=None):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # by default, we always use unstructured masking
        if mask_mode == 'unstructured':
            if self.complementary == False:
                a1, mask_a1, ids_restore_a1 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False)
                a2, mask_a2, ids_restore_a2 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False)
                a3, mask_a3, ids_restore_a3 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False)
            else:
                a1, mask_a1, ids_restore_a1, a2, mask_a2, ids_restore_a2, a3, mask_a3, ids_restore_a3 = self.random_masking_unstructured(a, mask_ratio_a, epoch_id, True, audio_mask=audio_mask, need_two=False, need_three=True)

            # p_x, vis_idx, mask_a2,  ids_restore_a2 = self.get_mask_a(a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
        else:
            a1, mask_a1, ids_restore_a1 = self.random_masking_unstructured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)
            # p_x2, vis_idx2, mask_a2, ids_restore_a2 = self.get_mask_a(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

        # visual branch always use unstructured masking

        if self.complementary == False:
            v1, mask_v1, ids_restore_v1 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False)
            v2, mask_v2, ids_restore_v2 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False)
            v3, mask_v3, ids_restore_v3 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False)
        else:
            v1, mask_v1, ids_restore_v1, v2, mask_v2, ids_restore_v2, v3, mask_v3, ids_restore_v3 = self.random_masking_unstructured(v, mask_ratio_v, epoch_id, False, video_mask=video_mask, need_two=False, need_three=True)
        
        # # print common elements between mask_a1 and mask_a2
        # matching_elements_count = 0
        # for m1, m2 in zip(mask_a1.flatten(), mask_a2.flatten()):
        #     if m1 == m2:
        #         matching_elements_count += 1

        # Process each masked audio and video input through their respective blocks
        for blk in self.blocks_a:
            a1 = blk(a1)
            a2 = blk(a2) 
            a3 = blk(a3)

        for blk in self.blocks_v:
            v1 = blk(v1)
            v2 = blk(v2) 
            v3 = blk(v3)
        x1 = torch.cat((a1, v1), dim=1)
        x2 = torch.cat((a2, v2), dim=1)
        x3 = torch.cat((a3, v3), dim=1)

        for blk in self.blocks_u:
            x1 = blk(x1)
            x2 = blk(x2)  # Again, consider cloning if needed
            x3 = blk(x3)
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        x3 = self.norm(x3)

        for blk in self.blocks_u:
            ca1 = blk(a1, 'a')
            ca2 = blk(a2, 'a')
            ca3 = blk(a3, 'a')
        ca1 = self.norm_a(ca1)
        ca2 = self.norm_a(ca2)
        ca3 = self.norm_a(ca3)

        for blk in self.blocks_u:
            cv1 = blk(v1, 'v')
            cv2 = blk(v2, 'v')
            cv3 = blk(v3, 'v')
        cv1 = self.norm_v(cv1)
        cv2 = self.norm_v(cv2)
        cv3 = self.norm_v(cv3)

        return x1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, ca1, cv1, x2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2, ca2, cv2, x3, mask_a3, ids_restore_a3, mask_v3, ids_restore_v3, ca3, cv3

    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):

        x = self.decoder_embed(x)
        # print("MASK SHAPE", mask_a.shape)
        # print("IDS SHAPE", ids_restore_a.shape)
        # append mask tokens to sequence
        # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
        mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
        a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  # no cls token
        a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # similar for the visual modality
        mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
        v_ = torch.cat([x[:, self.patch_embed_a.num_patches-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1)  # no cls token
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # concatenate audio and visual tokens
        x = torch.cat([a_, v_], dim=1) # Pass it separately
        # print("IN DECOER X SHAPE", x.shape)
        decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
        # print("DECODER SHAPE", decoder_pos_embed.shape)
        x = x + decoder_pos_embed

        # add modality indication tokens
        x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
        x[:, self.patch_embed_a.num_patches:, :] = x[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_a = self.decoder_pred_a(x[:, :self.patch_embed_a.num_patches, :])
        x_v = self.decoder_pred_v(x[:, self.patch_embed_a.num_patches:, :])

        # return audio and video tokens
        return x_a, x_v
    
    def forward_decoder_audio(self, x, mask_a, ids_restore_a):
        # Decoder embeddings
        x = self.decoder_embed_audio(x)

        # Append mask tokens for audio
        mask_tokens_a = self.mask_token_audio.repeat(x.shape[0], int(mask_a[0].sum()), 1)
        a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  
        a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Add position embeddings and modality indication for audio
        a_ = a_ + self.decoder_pos_embed_a
        a_ = a_ + self.decoder_modality_a

        # Apply Transformer blocks specialized for audio
        for blk in self.decoder_blocks_a:
            a_ = blk(a_)
        a_ = self.decoder_norm_a(a_)

        # Predictor projection for audio
        x_a = self.decoder_pred_a(a_)

        return x_a

    def forward_decoder_video(self, x, mask_v, ids_restore_v):
        # Decoder embeddings
        x = self.decoder_embed_video(x)

        # Append mask tokens for video
        mask_tokens_v = self.mask_token_video.repeat(x.shape[0], int(mask_v[0].sum()), 1)
        v_ = torch.cat([x[:, :self.patch_embed_v.num_patches-int(mask_v[0].sum()), :], mask_tokens_v], dim=1)  
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Add position embeddings and modality indication for video
        v_ = v_ + self.decoder_pos_embed_v
        v_ = v_ + self.decoder_modality_v

        # Apply Transformer blocks specialized for video
        for blk in self.decoder_blocks_v:
            v_ = blk(v_)
        v_ = self.decoder_norm_v(v_)

        # Predictor projection for video
        x_v = self.decoder_pred_v(v_)

        return x_v

    def forward_decoder_dual_mask(self, x1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, x2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2):
            
            x1 = self.decoder_embed(x1)
            x2 = self.decoder_embed(x2)
    
            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a1 = self.mask_token.repeat(x1.shape[0], int(mask_a1[0].sum()), 1)
            a1_ = torch.cat([x1[:, :self.patch_embed_a.num_patches-int(mask_a1[0].sum()), :], mask_tokens_a1], dim=1)  # no cls token
            a1_ = torch.gather(a1_, dim=1, index=ids_restore_a1.unsqueeze(-1).repeat(1, 1, x1.shape[2]))  # unshuffle

            # similar for the visual modality
            mask_tokens_v1 = self.mask_token.repeat(x1.shape[0], int(mask_v1[0].sum()), 1)
            v1_ = torch.cat([x1[:, self.patch_embed_a.num_patches-int(mask_a1[0].sum()):, :], mask_tokens_v1], dim=1)  # no cls token
            v1_ = torch.gather(v1_, dim=1, index=ids_restore_v1.unsqueeze(-1).repeat(1, 1, x1.shape[2]))  # unshuffle

            # concatenate audio and visual tokens
            x1 = torch.cat([a1_, v1_], dim=1) # Pass it separately

            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a2 = self.mask_token.repeat(x2.shape[0], int(mask_a2[0].sum()), 1)
            a2_ = torch.cat([x2[:, :self.patch_embed_a.num_patches-int(mask_a2[0].sum()), :], mask_tokens_a2], dim=1)  # no cls token
            a2_ = torch.gather(a2_, dim=1, index=ids_restore_a2.unsqueeze(-1).repeat(1, 1, x2.shape[2]))  # unshuffle

            # similar for the visual modality
            mask_tokens_v2 = self.mask_token.repeat(x2.shape[0], int(mask_v2[0].sum()), 1)
            v2_ = torch.cat([x2[:, self.patch_embed_a.num_patches-int(mask_a2[0].sum()):, :], mask_tokens_v2], dim=1)
            v2_ = torch.gather(v2_, dim=1, index=ids_restore_v2.unsqueeze(-1).repeat(1, 1, x2.shape[2]))

            # concatenate audio and visual tokens
            x2 = torch.cat([a2_, v2_], dim=1) # Pass it separately

            decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
            x1 = x1 + decoder_pos_embed
            x2 = x2 + decoder_pos_embed

            # add modality indication tokens
            x1[:, 0:self.patch_embed_a.num_patches, :] = x1[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
            x1[:, self.patch_embed_a.num_patches:, :] = x1[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

            x2[:, 0:self.patch_embed_a.num_patches, :] = x2[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
            x2[:, self.patch_embed_a.num_patches:, :] = x2[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x1 = blk(x1)
                x2 = blk(x2)
            x1 = self.decoder_norm(x1)
            x2 = self.decoder_norm(x2)

            # predictor projection
            x_a1 = self.decoder_pred_a(x1[:, :self.patch_embed_a.num_patches, :])
            x_v1 = self.decoder_pred_v(x1[:, self.patch_embed_a.num_patches:, :])

            x_a2 = self.decoder_pred_a(x2[:, :self.patch_embed_a.num_patches, :])
            x_v2 = self.decoder_pred_v(x2[:, self.patch_embed_a.num_patches:, :])

            # return audio and video tokens
            return x_a1, x_v1, x_a2, x_v2

    # Forward decoder triple mask
    def forward_decoder_triple_mask(self, x1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, x2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2, x3, mask_a3, ids_restore_a3, mask_v3, ids_restore_v3):
            
            x1 = self.decoder_embed(x1)
            x2 = self.decoder_embed(x2)
            x3 = self.decoder_embed(x3)

            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a1 = self.mask_token.repeat(x1.shape[0], int(mask_a1[0].sum()), 1)
            a1_ = torch.cat([x1[:, :self.patch_embed_a.num_patches-int(mask_a1[0].sum()), :], mask_tokens_a1], dim=1)  # no cls token
            a1_ = torch.gather(a1_, dim=1, index=ids_restore_a1.unsqueeze(-1).repeat(1, 1, x1.shape[2]))  # unshuffle

            # similar for the visual modality
            mask_tokens_v1 = self.mask_token.repeat(x1.shape[0], int(mask_v1[0].sum()), 1)
            v1_ = torch.cat([x1[:, self.patch_embed_a.num_patches-int(mask_a1[0].sum()):, :], mask_tokens_v1], dim=1)  # no cls token
            v1_ = torch.gather(v1_, dim=1, index=ids_restore_v1.unsqueeze(-1).repeat(1, 1, x1.shape[2]))  # unshuffle

            # concatenate audio and visual tokens
            x1 = torch.cat([a1_, v1_], dim=1) # Pass it separately

            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a2 = self.mask_token.repeat(x2.shape[0], int(mask_a2[0].sum()), 1)
            a2_ = torch.cat([x2[:, :self.patch_embed_a.num_patches-int(mask_a2[0].sum()), :], mask_tokens_a2], dim=1)  # no cls token
            a2_ = torch.gather(a2_, dim=1, index=ids_restore_a2.unsqueeze(-1).repeat(1, 1, x2.shape[2]))  # unshuffle

            # similar for the visual modality
            mask_tokens_v2 = self.mask_token.repeat(x2.shape[0], int(mask_v2[0].sum()), 1)
            v2_ = torch.cat([x2[:, self.patch_embed_a.num_patches-int(mask_a2[0].sum()):, :], mask_tokens_v2], dim=1)
            v2_ = torch.gather(v2_, dim=1, index=ids_restore_v2.unsqueeze(-1).repeat(1, 1, x2.shape[2]))

            # concatenate audio and visual tokens
            x2 = torch.cat([a2_, v2_], dim=1) # Pass it separately
            
            mask_tokens_a3 = self.mask_token.repeat(x3.shape[0], int(mask_a3[0].sum()), 1)
            a3_ = torch.cat([x3[:, :self.patch_embed_a.num_patches-int(mask_a3[0].sum()), :], mask_tokens_a3], dim=1)  # no cls token
            a3_ = torch.gather(a3_, dim=1, index=ids_restore_a3.unsqueeze(-1).repeat(1, 1, x3.shape[2]))  # unshuffle

            # similar for the visual modality
            mask_tokens_v3 = self.mask_token.repeat(x3.shape[0], int(mask_v3[0].sum()), 1)
            v3_ = torch.cat([x3[:, self.patch_embed_a.num_patches-int(mask_a3[0].sum()):, :], mask_tokens_v3], dim=1)
            v3_ = torch.gather(v3_, dim=1, index=ids_restore_v3.unsqueeze(-1).repeat(1, 1, x3.shape[2]))

            # concatenate audio and visual tokens
            x3 = torch.cat([a3_, v3_], dim=1) # Pass it separately
            
            decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
            x1 = x1 + decoder_pos_embed
            x2 = x2 + decoder_pos_embed
            x3 = x3 + decoder_pos_embed

            # add modality indication tokens
            x1[:, 0:self.patch_embed_a.num_patches, :] = x1[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
            x1[:, self.patch_embed_a.num_patches:, :] = x1[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

            x2[:, 0:self.patch_embed_a.num_patches, :] = x2[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
            x2[:, self.patch_embed_a.num_patches:, :] = x2[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

            x3[:, 0:self.patch_embed_a.num_patches, :] = x3[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
            x3[:, self.patch_embed_a.num_patches:, :] = x3[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x1 = blk(x1)
                x2 = blk(x2)
                x3 = blk(x3)
            x1 = self.decoder_norm(x1)
            x2 = self.decoder_norm(x2)
            x3 = self.decoder_norm(x3)

            # predictor projection
            x_a1 = self.decoder_pred_a(x1[:, :self.patch_embed_a.num_patches, :])
            x_v1 = self.decoder_pred_v(x1[:, self.patch_embed_a.num_patches:, :])

            x_a2 = self.decoder_pred_a(x2[:, :self.patch_embed_a.num_patches, :])
            x_v2 = self.decoder_pred_v(x2[:, self.patch_embed_a.num_patches:, :])
            
            x_a3 = self.decoder_pred_a(x3[:, :self.patch_embed_a.num_patches, :])
            x_v3 = self.decoder_pred_v(x3[:, self.patch_embed_a.num_patches:, :])

            # return audio and video tokens
            return x_a1, x_v1, x_a2, x_v2, x_a3, x_v3
    
    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() + 2*xy_kernel.mean()
        return mmd
    
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2)/dim*1.0)

    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation

        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == 'a':
            # Adjusting the shape for audio
            input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(input, 1, int(input.shape[2]/self.patch_embed_a.patch_size[0]), int(input.shape[3]/self.patch_embed_a.patch_size[1]), 16)
        elif modality == 'v':
            # Adjusting the shape for video
            target = self.patchify(input, 3, int(input.shape[2]/self.patch_embed_v.patch_size[0]), int(input.shape[3]/self.patch_embed_v.patch_size[1]), 16)

        if self.norm_pix_loss:
            # Normalizing the patches if needed
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # Calculating the squared error
        loss = (pred - target) ** 2

        # Average loss per patch
        loss_per_patch = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # Calculate the sum of loss per item in the batch, weighted by the mask
        loss_per_item = (loss_per_patch * mask).sum(dim=1) / mask.sum(dim=1)  # [N], mean loss per item in the batch

        # Calculate the total mean loss across the batch for all items
        total_mean_loss = loss_per_item.mean()

        # This returns both the total mean loss across the batch and the loss per item in the batch
        return total_mean_loss, loss_per_item

    def forward_PSNR(self, input_tensor, pred, mask, modality):
        psnr_metric = PeakSignalNoiseRatio()
        if modality == 'a':
            # for audio, need to adjust the shape
            input_tensor = input_tensor.unsqueeze(1)
            input_tensor = input_tensor.transpose(2, 3)
            target = self.patchify(input_tensor, 1, int(input_tensor.shape[2]/self.patch_embed_a.patch_size[0]), int(input_tensor.shape[3]/self.patch_embed_a.patch_size[1]), 16) # audio target shape = 1 * 16 * 16
            # [2, 512, 256]
            # Normalize each image in the batch separately BEFORE MASKING 
            # Create a copy of the target and prediction tensors 
            target_ = target.detach().clone()
            pred_ = pred.detach().clone()
            for i in range(target.shape[0]):
                # target[i] = (target[i] - target[i].min()) / (target[i].max() - target[i].min() + 1e-8)
                # pred[i] = (pred[i] - pred[i].min()) / (pred[i].max() - pred[i].min() + 1e-8)
                # using sigmoid
                target_[i] = torch.sigmoid(target_[i])
                pred_[i] = torch.sigmoid(pred_[i])
            target_ = self.unpatchify(target_, 1, int(input_tensor.shape[2]/self.patch_embed_a.patch_size[0]), int(input_tensor.shape[3]/self.patch_embed_a.patch_size[1]), 16)
            pred_ = self.unpatchify(pred_, 1, int(input_tensor.shape[2]/self.patch_embed_a.patch_size[0]), int(input_tensor.shape[3]/self.patch_embed_a.patch_size[1]), 16)
        elif modality == 'v':
            # print for patch embed 
            target = self.patchify(input_tensor, 3, int(input_tensor.shape[2]/self.patch_embed_v.patch_size[0]), int(input_tensor.shape[3]/self.patch_embed_v.patch_size[1]), 16) # video target shape = 3 * 16 * 16
            target_ = target.detach().clone()
            pred_ = pred.detach().clone()
            for i in range(target_.shape[0]):
                # target[i] = (target[i] - target[i].min()) / (target[i].max() - target[i].min() + 1e-8)
                # pred[i] = (pred[i] - pred[i].min()) / (pred[i].max() - pred[i].min() + 1e-8)
                # using sigmoid
                target_[i] = torch.sigmoid(target_[i])
                pred_[i] = torch.sigmoid(pred_[i])

            target_ = self.unpatchify(target_, 3, int(input_tensor.shape[2]/self.patch_embed_v.patch_size[0]), int(input_tensor.shape[3]/self.patch_embed_v.patch_size[1]), 16)
            pred_ = self.unpatchify(pred_, 3, int(input_tensor.shape[2]/self.patch_embed_v.patch_size[0]), int(input_tensor.shape[3]/self.patch_embed_v.patch_size[1]), 16)
        psnr_value = 0.0
        # Create a gaussian baseline to compare PSNR with
        for i in range(target.shape[0]):
            # print(target.shape)
            # input()
            target_ = target[i].detach().clone()
            pred_ = pred[i].detach().clone()
            psnr_metric.update(pred_[i], target_[i])
            # gaussian_tensor = torch.randn_like(pred[i])
            # psnr_metric.update(pred[i], gaussian_tensor)
            psnr_value += psnr_metric.compute()
            # if psnr value is nan, -inf, or inf, print values of target and prediction
            if torch.isnan(psnr_value) or torch.isinf(psnr_value):
                print("Target", target[i])
                print("Prediction", pred[i])
                print('psnr_value', psnr_value)
            # print(f'PSNR: {psnr_value}')
            psnr_metric.reset()
            
        psnr_value = psnr_value / target.shape[0]
        return psnr_value

    def forward(self, epoch_id, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, kld_loss_weight=10, mask_mode='unstructured', labels=None, audio_mask=None,video_mask=None):
        kld_weight = self.kd_weight
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        # weights = torch.stack([
        #     # self.log_noise_a,
        #     # self.log_noise_v, 
        #     self.log_noise_c, 
        #     self.log_noise_AA, 
        #     self.log_noise_VV, 
        #     self.log_noise_AV
        # ]).squeeze() 
        # softmax_weights = F.softmax(weights, dim=0) * 4

        if self.dual_mask == True:
            latent1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, latent_c_a1, latent_c_v1, latent2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2, latent_c_a2, latent_c_v2 = self.forward_encoder_dual_mask(epoch_id, audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode, video_mask=video_mask, audio_mask=audio_mask)
            # decoder preds

            pred_a1, pred_v1, pred_a2, pred_v2 = self.forward_decoder_dual_mask(latent1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, latent2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2)
            # if mae loss is used         
            if mae_loss_weight != 0:
                loss_mae_a1, loss_per_item_mae_a1 = self.forward_mae_loss(audio, pred_a1, mask_a1, 'a')
                loss_mae_v1, loss_per_item_mae_v1 = self.forward_mae_loss(imgs, pred_v1, mask_v1, 'v')
                loss_mae_a2, loss_per_item_mae_a2 = self.forward_mae_loss(audio, pred_a2, mask_a2, 'a')
                loss_mae_v2, loss_per_item_mae_v2 = self.forward_mae_loss(imgs, pred_v2, mask_v2, 'v')


                loss_mae1 = mae_loss_weight * (loss_mae_a1 + loss_mae_v1)
                loss_mae2 = mae_loss_weight * (loss_mae_a2 + loss_mae_v2)

            else:
                loss_mae_a1, loss_mae_v1, loss_mae1 = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)
                loss_mae_a2, loss_mae_v2, loss_mae2 = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

            if contrast_loss_weight != 0:
                loss_c1, c_acc1 = self.forward_contrastive(latent_c_a1.mean(dim=1), latent_c_v1.mean(dim=1))
                loss_c2, c_acc2 = self.forward_contrastive(latent_c_a2.mean(dim=1), latent_c_v2.mean(dim=1))

                loss_c1 = contrast_loss_weight * loss_c1
                loss_c2 = contrast_loss_weight * loss_c2
            
            # Kl divergence loss between latent1 and latent2
            if kld_loss_weight != 0:
                if self.AAVV == True:
                    latent1_mean = latent1.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent2_mean = latent2.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent_c_a1_mean = latent_c_a1.mean(dim=1)
                    latent_c_v1_mean = latent_c_v1.mean(dim=1)
                    latent_c_a2_mean = latent_c_a2.mean(dim=1)
                    latent_c_v2_mean = latent_c_v2.mean(dim=1)

                    latent1_positive = latent1_mean + abs(latent1_mean.min())
                    latent2_positive = latent2_mean + abs(latent2_mean.min())
                    latent1_normalized = latent1_positive / latent1_positive.sum(dim=1, keepdim=True)
                    latent2_normalized = latent2_positive / latent2_positive.sum(dim=1, keepdim=True)

                    latent_c_a1_positive = latent_c_a1_mean + abs(latent_c_a1_mean.min())
                    latent_c_v1_positive = latent_c_v1_mean + abs(latent_c_v1_mean.min())
                    latent_c_a1_normalized = latent_c_a1_positive / latent_c_a1_positive.sum(dim=1, keepdim=True)
                    latent_c_v1_normalized = latent_c_v1_positive / latent_c_v1_positive.sum(dim=1, keepdim=True)

                    latent_c_a2_positive = latent_c_a2_mean + abs(latent_c_a2_mean.min())
                    latent_c_v2_positive = latent_c_v2_mean + abs(latent_c_v2_mean.min())
                    latent_c_a2_normalized = latent_c_a2_positive / latent_c_a2_positive.sum(dim=1, keepdim=True)
                    latent_c_v2_normalized = latent_c_v2_positive / latent_c_v2_positive.sum(dim=1, keepdim=True)

                    # use js divergence between latent1 and latent2
                    js_loss = js_divergence(latent1_normalized, latent2_normalized)
                    js_loss_v = js_divergence(latent_c_v1_normalized, latent_c_v2_normalized)
                    js_loss_a = js_divergence(latent_c_a1_normalized, latent_c_a2_normalized) 
                    # Average the KL divergence loss over all items in the batch
                    # if self.dynamic_weighting == True:
                    #     loss_kd = kld_loss_weight * (js_loss_a * softmax_weights[1] * js_loss_v * softmax_weights[2] * js_loss * softmax_weights[3])  
                    
                    # else: 
                    loss_kd = kld_loss_weight * (js_loss + js_loss_v + js_loss_a)
                   
                else:

                    # mean across patches
                    latent1_mean = latent1.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent2_mean = latent2.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    
                    latent1_positive = latent1_mean + abs(latent1_mean.min())
                    latent2_positive = latent2_mean + abs(latent2_mean.min())
                    latent1_normalized = latent1_positive / latent1_positive.sum(dim=1, keepdim=True)
                    latent2_normalized = latent2_positive / latent2_positive.sum(dim=1, keepdim=True)

                    # use js divergence between latent1 and latent2
                    # js_loss_dummy = js_divergence(latent1_normalized, latent2_normalized)
                    # print(js_loss_dummy.shape)
                    # resistor_parallel = False
                    # if resistor_parallel == True:
                    #     kl_loss1 = kl_divergence(latent1_normalized, latent2_normalized)
                    #     kl_loss2 = kl_divergence(latent2_normalized, latent1_normalized)
                    #     resistor_parallel_loss = kl_loss1 * kl_loss2 / (kl_loss1 + kl_loss2)
                    #     loss_kd = kld_loss_weight * resistor_parallel_loss
                    # Average the KL divergence loss over all items in the batch
                    # if self.dynamic_weighting == True:
                    #     js_loss = js_divergence(latent1_normalized, latent2_normalized)
                    #     loss_kd = kld_loss_weight * js_loss * softmax_weights[3]
                    # else:
                    js_loss = js_divergence(latent1_normalized, latent2_normalized)
                    loss_kd = kld_loss_weight * js_loss
                    # print("SHAPE", loss_kd.shape)
                    # print("LOSS", loss_kd)  
            else:
                loss_kd = torch.tensor(0.0, device=audio.device)

            # if self.dynamic_weighting == True:
            #     if self.dynamic_weight_normalization_method == 'total_sum1':
            #         pass
            #     # Squash each individual weight to be between 0 and 1
            #     elif self.dynamic_weight_normalization_method == 'individual_sum1':
            #         weight_mae = torch.sigmoid(1/2 * torch.exp(self.log_noise_mae))
            #         weight_c = torch.sigmoid(1/2 * torch.exp(self.log_noise_c))
            #         weight_latent = torch.sigmoid(1/2 * torch.exp(self.log_noise_latent))
            #         weight_a = torch.sigmoid(1/2 * torch.exp(self.log_noise_a))
            #         weight_v = torch.sigmoid(1/2 * torch.exp(self.log_noise_v))

            #     # No normalization of dynamic weights
            #     elif self.dynamic_weight_normalization_method == 'unormalized':
            #         weight_mae = 1 / (2 * torch.exp(self.log_noise_mae))
            #         weight_c = 1 / (2 * torch.exp(self.log_noise_c))
            #         weight_latent = 1 / (2 * torch.exp(self.log_noise_latent))
            #         weight_a = 1 / (2 * torch.exp(self.log_noise_a))
            #         weight_v = 1 / (2 * torch.exp(self.log_noise_v))
            #     else:
            #         raise ValueError(f"Invalid normalization method: {self.dynamic_weight_normalization_method}")      
            #     # Compute the weighted loss
            #     loss = weight_mae * (loss_mae1 + loss_mae2) + weight_c * (loss_c1 + loss_c2) + weight_latent * loss_kd + self.log_noise_mae + self.log_noise_c + self.log_noise_latent
            #     # print weights
            #     print("MAE WEIGHT", weight_mae)
            #     print("C WEIGHT", weight_c)
            #     print("LATENT WEIGHT", weight_latent)            
            # else:
            
            if self.dynamic_weighting == True:
                if self.AAVV == True:
                    loss_mae = loss_mae1 + loss_mae2 
                    loss_c = loss_c1 + loss_c2
                    losses_tensor = torch.tensor([loss_mae.sum(), loss_c.sum(), js_loss_a.sum(), js_loss_v.sum(), js_loss.sum()], requires_grad=True).to(audio.device)
                    logits = self.dw_fc(losses_tensor)
                    weights = F.softmax(logits, dim=0) * 4
                    loss = loss_mae1 + loss_mae2 + weights[0] * (loss_c1 + loss_c2) + weights[1] * js_loss_a + weights[2] * js_loss_v + weights[3] * js_loss
                else:
                    loss_mae = loss_mae1 + loss_mae2 
                    loss_c = loss_c1 + loss_c2
                    losses_tensor = torch.tensor([loss_mae.sum(), loss_c.sum(), loss_kd.sum()], requires_grad=True).to(audio.device)
                    losses_tensor = F.relu(self.dw_fc1(losses_tensor))
                    losses_tensor = F.relu(self.dw_fc2(losses_tensor))
                    losses_tensor = F.relu(self.dw_fc3(losses_tensor))
                    logits = self.dw_fc4(losses_tensor)
                    weights = torch.exp(F.softmax(logits, dim=0) * 3)
                    loss =  (loss_mae1 + loss_mae2) * weights[0] + weights[1] * (loss_c1 + loss_c2) + weights[2] * loss_kd
                print("WEIGHTS", weights)
            else:
                loss = loss_mae1 + loss_mae2 + loss_c1 + loss_c2 + loss_kd
            return loss, loss_mae1, loss_mae_a1, loss_mae_v1, loss_c1, mask_a1, mask_v1, c_acc1, loss_mae2, loss_mae_a2, loss_mae_v2, loss_c2, mask_a2, mask_v2, c_acc2, loss_kd
        
        if self.triple_mask == True:
            latent1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, latent_c_a1, latent_c_v1, latent2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2, latent_c_a2, latent_c_v2, latent3, mask_a3, ids_restore_a3, mask_v3, ids_restore_v3, latent_c_a3, latent_c_v3 = self.forward_encoder_triple_mask(epoch_id, audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode, video_mask=video_mask, audio_mask=audio_mask)
            # decoder preds
  
            pred_a1, pred_v1, pred_a2, pred_v2, pred_a3, pred_v3 = self.forward_decoder_triple_mask(latent1, mask_a1, ids_restore_a1, mask_v1, ids_restore_v1, latent2, mask_a2, ids_restore_a2, mask_v2, ids_restore_v2, latent3, mask_a3, ids_restore_a3, mask_v3, ids_restore_v3)
            # if mae loss is used         
            if mae_loss_weight != 0:
                loss_mae_a1, loss_per_item_mae_a1 = self.forward_mae_loss(audio, pred_a1, mask_a1, 'a')
                loss_mae_v1, loss_per_item_mae_v1 = self.forward_mae_loss(imgs, pred_v1, mask_v1, 'v')
                loss_mae_a2, loss_per_item_mae_a2 = self.forward_mae_loss(audio, pred_a2, mask_a2, 'a')
                loss_mae_v2, loss_per_item_mae_v2 = self.forward_mae_loss(imgs, pred_v2, mask_v2, 'v')
                loss_mae_a3, loss_per_item_mae_a3 = self.forward_mae_loss(audio, pred_a3, mask_a3, 'a')
                loss_mae_v3, loss_per_item_mae_v3 = self.forward_mae_loss(imgs, pred_v3, mask_v3, 'v')

                if self.dynamic_weighting == True:
                    # loss_mae_a1 = softmax_weights[0] * loss_mae_a1
                    # loss_mae_a2 = softmax_weights[0] * loss_mae_a1
                    # loss_mae_v1 = softmax_weights[1] * loss_mae_v1
                    # loss_mae_v2 = softmax_weights[1] * loss_mae_v2
                    loss_mae1 = (loss_mae_a1 + loss_mae_v1) * mae_loss_weight
                    loss_mae2 = (loss_mae_a2 + loss_mae_v2) * mae_loss_weight
                    loss_mae3 = (loss_mae_a3 + loss_mae_v3) * mae_loss_weight
                else:
                    loss_mae1 = mae_loss_weight * (loss_mae_a1 + loss_mae_v1)
                    loss_mae2 = mae_loss_weight * (loss_mae_a2 + loss_mae_v2)
                    loss_mae3 = mae_loss_weight * (loss_mae_a3 + loss_mae_v3)

            else:
                loss_mae_a1, loss_mae_v1, loss_mae1 = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)
                loss_mae_a2, loss_mae_v2, loss_mae2 = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)
                loss_mae_a3, loss_mae_v3, loss_mae3 = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

            if contrast_loss_weight != 0:
                loss_c1, c_acc1 = self.forward_contrastive(latent_c_a1.mean(dim=1), latent_c_v1.mean(dim=1))
                loss_c2, c_acc2 = self.forward_contrastive(latent_c_a2.mean(dim=1), latent_c_v2.mean(dim=1))
                loss_c3, c_acc3 = self.forward_contrastive(latent_c_a3.mean(dim=1), latent_c_v3.mean(dim=1))


                loss_c1 = contrast_loss_weight * loss_c1
                loss_c2 = contrast_loss_weight * loss_c2
                loss_c3 = contrast_loss_weight * loss_c3

            # Kl divergence loss between latent1 and latent2
            if kld_loss_weight != 0:
                if self.AAVV == True:
                    latent1_mean = latent1.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent2_mean = latent2.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent_c_a1_mean = latent_c_a1.mean(dim=1)
                    latent_c_v1_mean = latent_c_v1.mean(dim=1)
                    latent_c_a2_mean = latent_c_a2.mean(dim=1)
                    latent_c_v2_mean = latent_c_v2.mean(dim=1)

                    latent1_positive = latent1_mean + abs(latent1_mean.min())
                    latent2_positive = latent2_mean + abs(latent2_mean.min())
                    latent1_normalized = latent1_positive / latent1_positive.sum(dim=1, keepdim=True)
                    latent2_normalized = latent2_positive / latent2_positive.sum(dim=1, keepdim=True)

                    latent_c_a1_positive = latent_c_a1_mean + abs(latent_c_a1_mean.min())
                    latent_c_v1_positive = latent_c_v1_mean + abs(latent_c_v1_mean.min())
                    latent_c_a1_normalized = latent_c_a1_positive / latent_c_a1_positive.sum(dim=1, keepdim=True)
                    latent_c_v1_normalized = latent_c_v1_positive / latent_c_v1_positive.sum(dim=1, keepdim=True)

                    latent_c_a2_positive = latent_c_a2_mean + abs(latent_c_a2_mean.min())
                    latent_c_v2_positive = latent_c_v2_mean + abs(latent_c_v2_mean.min())
                    latent_c_a2_normalized = latent_c_a2_positive / latent_c_a2_positive.sum(dim=1, keepdim=True)
                    latent_c_v2_normalized = latent_c_v2_positive / latent_c_v2_positive.sum(dim=1, keepdim=True)

                    # use js divergence between latent1 and latent2
                    js_loss = js_divergence(latent1_normalized, latent2_normalized)
                    js_loss_v = js_divergence(latent_c_v1_normalized, latent_c_v2_normalized)
                    js_loss_a = js_divergence(latent_c_a1_normalized, latent_c_a2_normalized)
                    # print("JS_LOSS", js_loss)
                    # print("JS_LOSS_V", js_loss_v)
                    # print("JS_LOSS_A", js_loss_a)
                    # Average the KL divergence loss over all items in the batch
                    # if self.dynamic_weighting == True:
                    #     js_loss = softmax_weights[3] * js_loss.sum() * kld_loss_weight
                    #     js_loss_v = softmax_weights[2] * js_loss_v.sum() * kld_loss_weight
                    #     js_loss_a = softmax_weights[1] * js_loss_a.sum() * kld_loss_weight
                    #     loss_kd = js_loss + js_loss_v + js_loss_a
                    # else:
                    loss_kd = kld_loss_weight * (js_loss + js_loss_v + js_loss_a)
                    
                else:
                    # latent1 and latent2 have shapes [batch_size, 177, 768]
                    # Calculate combined per item reconstruction losses for both sets

                    # mean across patches
                    latent1_mean = latent1.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent2_mean = latent2.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]
                    latent3_mean = latent3.mean(dim=1)  # Averaging across the patch dimension [batch_size, 768]

                    latent1_positive = latent1_mean + abs(latent1_mean.min())
                    latent2_positive = latent2_mean + abs(latent2_mean.min())
                    latent3_positive = latent3_mean + abs(latent3_mean.min())

                    latent1_normalized = latent1_positive / latent1_positive.sum(dim=1, keepdim=True)
                    latent2_normalized = latent2_positive / latent2_positive.sum(dim=1, keepdim=True)
                    latent3_normalized = latent3_positive / latent3_positive.sum(dim=1, keepdim=True)

                    # use js divergence between latent1 and latent2
                    # js_loss_dummy = js_divergence(latent1_normalized, latent2_normalized)
                    # print(js_loss_dummy.shape)
                    # resistor_parallel = False
                    # if resistor_parallel == True:
                    #     kl_loss12_1 = kl_divergence(latent1_normalized, latent2_normalized)
                    #     kl_loss12_2 = kl_divergence(latent2_normalized, latent1_normalized)
                    #     kl_loss13_1 = kl_divergence(latent1_normalized, latent3_normalized)
                    #     kl_loss13_2 = kl_divergence(latent3_normalized, latent1_normalized)
                    #     kl_loss23_1 = kl_divergence(latent2_normalized, latent3_normalized)
                    #     kl_loss23_2 = kl_divergence(latent3_normalized, latent2_normalized)
                    #     # resistor_parallel_loss = kl_loss1 * kl_loss2 / (kl_loss1 + kl_loss2)
                    #     # loss_kd = kld_loss_weight * resistor_parallel_loss
                    #     print("DONT USE")
                    # # Average the KL divergence loss over all items in the batch
                    # else:
                    js_loss12 = js_divergence(latent1_normalized, latent2_normalized)
                    js_loss13 = js_divergence(latent1_normalized, latent3_normalized)
                    js_loss23 = js_divergence(latent2_normalized, latent3_normalized)
                    js_loss = (js_loss12 + js_loss13 + js_loss23)/3
                    loss_kd = kld_loss_weight * js_loss
                    # print("SHAPE", loss_kd.shape)
                    # print("LOSS", loss_kd)  
            else:
                loss_kd = torch.tensor(0.0, device=audio.device)

            if self.dynamic_weighting == True:
                if self.dynamic_weight_normalization_method == 'total_sum1':
                    pass
                # Squash each individual weight to be between 0 and 1
                elif self.dynamic_weight_normalization_method == 'individual_sum1':
                    weight_mae = torch.sigmoid(1/2 * torch.exp(self.log_noise_mae))
                    weight_c = torch.sigmoid(1/2 * torch.exp(self.log_noise_c))
                    weight_latent = torch.sigmoid(1/2 * torch.exp(self.log_noise_latent))
                    weight_a = torch.sigmoid(1/2 * torch.exp(self.log_noise_a))
                    weight_v = torch.sigmoid(1/2 * torch.exp(self.log_noise_v))

                # No normalization of dynamic weights
                elif self.dynamic_weight_normalization_method == 'unormalized':
                    weight_mae = 1 / (1 * torch.exp(self.log_noise_mae))
                    weight_c = 1 / (1 * torch.exp(self.log_noise_c))
                    weight_latent = 1 / (1 * torch.exp(self.log_noise_latent))
                    weight_AA = 1 / (1 * torch.exp(self.log_noise_AA))
                    weight_VV = 1 / (1 * torch.exp(self.log_noise_VV))
                    weight_AV = 1 / (1 * torch.exp(self.log_noise_AV))

                    weight_a = 1 / (2 * torch.exp(self.log_noise_a))
                    weight_v = 1 / (2 * torch.exp(self.log_noise_v))
                else:
                    raise ValueError(f"Invalid normalization method: {self.dynamic_weight_normalization_method}")
            else:  
                loss = loss_mae1 + loss_mae2 + loss_mae3 + loss_c1 + loss_c2 + loss_c3 + loss_kd

            return loss, loss_mae1, loss_mae_a1, loss_mae_v1, loss_c1, mask_a1, mask_v1, c_acc1, loss_mae2, loss_mae_a2, loss_mae_v2, loss_c2, mask_a2, mask_v2, c_acc2, loss_mae3, loss_mae_a3, loss_mae_v3, loss_c3, mask_a3, mask_v3, c_acc3, loss_kd
    
        else:
            latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v  = self.forward_encoder(epoch_id, audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode, video_mask=video_mask, audio_mask=audio_mask)
            
            if self.split_decoder == True:
                pred_a = self.forward_decoder_audio(latent, mask_a, ids_restore_a)
                pred_v = self.forward_decoder_video(latent, mask_v, ids_restore_v)
            else:
                pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)

            #print(latent_c_a.shape)
            #print(latent_c_v.shape)

            initial_latent_loss_weight = 0.1
            weighted_latent_mse = False
            if weighted_latent_mse == True:
                weights = torch.arange(1, 769).float()  # 1, 2, 3, ..., 768
                loss_latent = torch.mean((weights * initial_latent_loss_weight * latent_c_a.mean(dim=1) - latent_c_v.mean(dim=1))**2) 
            else:
                loss_latent = torch.mean((latent_c_a.mean(dim=1) - latent_c_v.mean(dim=1))**2)
            
            # if mae loss is used
            if mae_loss_weight != 0:
            
                loss_mae_a, _ = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
                loss_mae_v, _ = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
                loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
            else:
                loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        # if contrastive loss is used
            if contrast_loss_weight != 0:
                # note this is single directional
                loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1))
                loss_c = contrast_loss_weight * loss_c
            else:
                loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

            if self.knowledge_distillation == True:
                # get psnr value
                psnr_a = self.forward_PSNR(audio, pred_a, mask_a, 'a')
                psnr_v = self.forward_PSNR(imgs, pred_v, mask_v, 'v')
                k = self.k_value
                if epoch_id == 0:
                    self.psnr_a_prev = psnr_a
                    self.psnr_v_prev = psnr_v
                elif epoch_id >= 1:
                    self.delta_psnr_a = psnr_a - self.psnr_a_prev
                    self.delta_psnr_v = psnr_v - self.psnr_v_prev
                    self.psnr_a_prev = psnr_a
                    self.psnr_v_prev = psnr_v
                # 1 Determine student and teacher based on delta psnr
                if self.delta_psnr_a >= self.delta_psnr_v:
                    student_loss = loss_mae_v
                    teacher_loss = loss_mae_a
                else:
                    student_loss = loss_mae_a
                    teacher_loss = loss_mae_v
                # 2. Compute the Scaling Factor for Student's Gradients
                if student_loss > teacher_loss:
                    scaling_factor = torch.exp(k * (student_loss - teacher_loss) / student_loss) * student_loss
                else:
                    scaling_factor = torch.exp(student_loss - teacher_loss) * student_loss
                
                student_loss = student_loss * scaling_factor
                new_loss_mae = student_loss + teacher_loss
            else:
                new_loss_mae = loss_mae_a + loss_mae_v

            if self.dynamic_weighting == True:
                if self.dynamic_weight_normalization_method == 'total_sum1':
                    pass
                # Squash each individual weight to be between 0 and 1
                elif self.dynamic_weight_normalization_method == 'individual_sum1':
                    weight_mae = torch.sigmoid(1/2 * torch.exp(self.log_noise_mae))
                    weight_c = torch.sigmoid(1/2 * torch.exp(self.log_noise_c))
                    weight_latent = torch.sigmoid(1/2 * torch.exp(self.log_noise_latent))
                    weight_a = torch.sigmoid(1/2 * torch.exp(self.log_noise_a))
                    weight_v = torch.sigmoid(1/2 * torch.exp(self.log_noise_v))

                # No normalization of dynamic weights
                elif self.dynamic_weight_normalization_method == 'unormalized':
                    weight_mae = 1 / (2 * torch.exp(self.log_noise_mae))
                    weight_c = 1 / (2 * torch.exp(self.log_noise_c))
                    weight_latent = 1 / (2 * torch.exp(self.log_noise_latent))
                    weight_a = 1 / (2 * torch.exp(self.log_noise_a))
                    weight_v = 1 / (2 * torch.exp(self.log_noise_v))
                else:
                    raise ValueError(f"Invalid normalization method: {self.dynamic_weight_normalization_method}")      
                
            else:
                weight_a = 1
                weight_v = 1
                weight_mae = 1
                weight_c = 1
                weight_latent = 1

            if self.model == 'vanilla':
                if self.dynamic_weighting:
                    loss = weight_mae * new_loss_mae + weight_c * loss_c + self.log_noise_mae + self.log_noise_c
                else:
                    loss = new_loss_mae + loss_c
            elif self.model == 'latent':
                if self.dynamic_weighting == True:
                    loss = weight_mae * new_loss_mae + weight_c * loss_c + weight_latent * loss_latent + self.log_noise_mae + self.log_noise_c + self.log_noise_latent
                else:
                    loss = new_loss_mae + loss_c + loss_latent
            else:
                raise ValueError(f"Invalid model type: {self.model}")
            return loss, new_loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc

    # used only for inpainting, ignore if inpainting is not of interest
    def forward_inpaint(self, epoch_id, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mask_mode='unstructured'):
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(0, audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)  # [N, L, p*p*3]
        loss_pixel_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
        loss_pixel_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
        return pred_a, pred_v, mask_a, mask_v, loss_pixel_a, loss_pixel_v

    # used for retrieval, ignore if retrieval is not of interest
    def forward_feat(self, a, v):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # the modality-specific stream
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        # use modality specific normalization,
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)

        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_v(v)
        return a, v

# the finetuned CAV-MAE model
class CAVMAEFT(nn.Module):
    def __init__(
                self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                tr_pos=True, # CHECK IF WE NEED THIS CUZ IT WASN'T BEING PASSED IN OG FINETUNE ARGS
                # Other model parameters
                dynamic_weighting=False, 
                model_type='vanilla',
                dynamic_weight_normalization_method='unormalized',
                knowledge_distillation=False, 
                k_value=-0.25, 
                absolute_noise=False,
                kld_mlp = False,
            ):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        print('Use norm_pix_loss: ', norm_pix_loss)

        self.model_type = model_type
        self.dynamic_weight_normalization_method = dynamic_weight_normalization_method
        self.knowledge_distillation = knowledge_distillation
        self.k_value = k_value
        self.absolute_noise = absolute_noise
        self.dynamic_weighting = dynamic_weighting
        self.kld_mlp = kld_mlp
        print('Model Type: ', self.model_type)
        print('Dynamic Weight Normalization Method: ', self.dynamic_weight_normalization_method)
        print('Knowledge Distillation: ', self.knowledge_distillation)
        print('K Value: ', self.k_value)
        print('Absolute Noise: ', self.absolute_noise)
        print('Dynamic Weighting: ', self.dynamic_weighting)

        self.log_noise_latent = nn.Parameter(torch.zeros(1, device='cuda'), requires_grad=True)
        self.log_noise_classification = nn.Parameter(torch.zeros(1, device='cuda'), requires_grad=True)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(12 - modality_specific_depth)])

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        if self.kld_mlp:
            # mlp with input dimension is (B, 177, 768) to output dimension (B, 177, 512) to (B, 177, 256)
            self.kld_mlp = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )
            self.mlp_head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, label_dim))

        else:
            self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))
        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode, audio_mask=None,video_mask=None, labels=None, loss_fn=None):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            for blk in self.blocks_u:
                ca = blk(a, 'a')
            for blk in self.blocks_u:
                cv = blk(v, 'v')
            
            if self.knowledge_distillation == True:
                # Latent MSE loss between audio and video embeddings
                loss_latent = torch.mean((ca.mean(dim=1) - cv.mean(dim=1))**2)
            else:
                loss_latent = torch.tensor(0.0, device=a.device) 
            x = x.mean(dim=1)
            if self.kld_mlp:
                x = self.kld_mlp(x)
            x = self.mlp_head(x)
            
            classification_loss = loss_fn(x, labels)
            if self.dynamic_weighting:
                if self.model_type == 'latent':
                    weight_latent = 1 / (2 * torch.exp(self.log_noise_latent))
                    weight_classification = 1 / (2 * torch.exp(self.log_noise_classification))
                    loss = weight_latent * loss_latent + weight_classification * classification_loss + self.log_noise_latent + self.log_noise_classification
            else:
                if self.model_type == 'latent':
                    loss = classification_loss + loss_latent
                elif self.model_type == 'vanilla':
                    loss = classification_loss

            return x, loss

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            classification_loss = loss_fn(x, labels)
            return x, classification_loss

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            classification_loss = loss_fn(x, labels)
            return x, classification_loss

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a') # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v') # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # for retrieval
    def forward_feat(self, a, v, mode='av'):
        # return both audio and visual
        if mode == 'av':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, 'v')

            v = self.norm_v(v)
            return a, v

        # return only audio
        if mode == 'a':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_u:
                a = blk(a, 'a')

            a = self.norm_a(a)
            return a