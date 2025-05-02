import logging
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple
from collections import deque

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import einops
import torch.optim as optim
import wandb
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from agents.models.beso.models.edm_diffusion.gc_sampling import *
from agents.models.beso.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from mode.utils.lang_buffer import AdvancedLangEmbeddingBuffer

from mode.callbacks.ema import EMA

logger = logging.getLogger(__name__)


class BesoAgent(pl.LightningModule):
    def __init__(
            self,
            model: DictConfig,
            language_encoders: DictConfig,
            obs_encoders: DictConfig,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim,
            action_dim,
            decay: float,
            obs_seq_len: int,
            act_seq_len: int,
            state_dim = 7,
            use_lr_scheduler: bool = True,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            if_film_condition: bool = False,
            if_robot_states: bool = False,
            use_text_not_embedding: bool = True,
            ckpt_path=None,
    ):
        super().__init__()

        self.working_dir = os.getcwd()
        self.scaler = None

        self.model = hydra.utils.instantiate(model)

        # Initialize model and encoder
        self.img_encoder = hydra.utils.instantiate(obs_encoders)
        self.language_encoder = hydra.utils.instantiate(language_encoders)
        self.state_emb = nn.Linear(state_dim, latent_dim)

        # for inference
        self.rollout_step_counter = 0
        self.act_seq_len = act_seq_len
        self.obs_seq_len = obs_seq_len

        self.action_dim = action_dim

        self.use_lr_scheduler = use_lr_scheduler

        self.latent_dim = latent_dim
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.modality_scope = "lang"

        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type

        self.if_film_condition = if_film_condition
        self.if_robot_states = if_robot_states

        self.decay = decay

        self.state_recons = False
        self.use_text_not_embedding = use_text_not_embedding

        self.lang_buffer = AdvancedLangEmbeddingBuffer(self.language_encoder, 10000)

        if ckpt_path is not None:
            self.load_pretrained_model(ckpt_path)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''

        optim_groups = self.get_optim_groups()

        optim_groups.extend([
            {"params": self.img_encoder.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])

        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate, betas=self.optimizer_config.betas)
        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def get_optim_groups(self):
        # Helper function to check if a parameter should use weight decay
        def use_weight_decay(name):
            return all(x not in name for x in ['bias', 'LayerNorm', 'embedding'])

        # Split parameters into two groups
        decay = []
        no_decay = []

        for name, param in self.model.inner_model.named_parameters():
            if use_weight_decay(name):
                decay.append(param)
            else:
                no_decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        return optim_groups

    def compute_input_embeddings(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        latent_goal = None
        # last images are the randomly sampled future goal images for models learned with image goals

        obs_dict = {
            'rgb_static': einops.rearrange(dataset_batch["rgb_obs"]['rgb_static'], 'b t c h w -> (b t) c h w'),
            'rgb_gripper': einops.rearrange(dataset_batch["rgb_obs"]['rgb_gripper'], 'b t c h w -> (b t) c h w'),
            'pc_static': einops.rearrange(dataset_batch["depth_obs"]['depth_static'], 'b t c h w -> (b t) c h w'),
            'pc_gripper': einops.rearrange(dataset_batch["depth_obs"]['depth_gripper'], 'b t c h w -> (b t) c h w')
        }

        if self.use_text_not_embedding:
            # latent_goal = self.language_encoder(dataset_batch["lang_text"]).to(rgb_static.dtype)
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(dataset_batch["lang_text"]).to(
                obs_dict['rgb_static'].dtype)
        else:
            latent_goal = self.language_encoder(dataset_batch["lang"]).to(obs_dict['rgb_static'].dtype)

        obs_dict['latent_goal'] = latent_goal

        perceptual_emb = self.img_encoder(obs_dict)

        if self.if_robot_states:
            perceptual_emb['robot_obs'] = dataset_batch['robot_obs']

        return perceptual_emb, latent_goal

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Compute and return the training loss for the mode Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch.

        Returns:
            loss tensor
        """
        action_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0
        batch_sizes = []
        for self.modality_scope, dataset_batch in batch.items():
            # Compute the required embeddings
            perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)

            act_loss = self.diffusion_loss(
                perceptual_emb,
                dataset_batch["actions"],
                latent_goal
            )

            action_loss += act_loss

            batch_sizes.append(dataset_batch["actions"].shape[0])
            total_bs += dataset_batch["actions"].shape[0]

        # Average losses across datasets
        batch_len = len(batch)
        action_loss = action_loss / batch_len
        # Log the metrics
        self._log_training_metrics(action_loss, total_bs)

        return act_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> None:  # Change return type to None
        output = {}
        dataset_batch = batch
        perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)

        action_pred = self.denoise_actions(
            perceptual_emb,
            latent_goal,
            inference=True,
        )

        actions = dataset_batch["actions"].to(self.device)
        pred_loss = torch.nn.functional.mse_loss(action_pred, actions)

        self._log_validation_metrics(pred_loss)

        output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
        output["validation_loss"] = pred_loss
        return output

    def log_with_device_check(self, name, value, **kwargs):
        if torch.is_tensor(value) and value.device.type != "cuda":
            value = value.to(self.device)
        self.log(name, value, **kwargs)

    def _log_validation_metrics(self, pred_loss):
        """
        Log the validation metrics.
        """
        self.log(f"val_act/{self.modality_scope}_act_loss_pp", pred_loss, sync_dist=True)

    def diffusion_loss(
            self,
            perceptual_emb: torch.Tensor,
            actions: torch.Tensor,
            latent_goal: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)

        return loss

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")
        # self.model.inner_model.reset_expert_caches() if hasattr(self.model.inner_model, 'reset_expert_caches') else None
        # self.need_precompute_experts_for_inference = True

    def denoise_actions(  # type: ignore
            self,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor = None,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10

        # if len(latent_goal.shape) < len(
        #         perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
        #     latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)

        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)

        x = torch.randn((len(perceptual_emb), self.act_seq_len, self.action_dim), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, self.sampler_type, extra_args)

        return actions

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def _log_training_metrics(self, action_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps to save computation and increase efficiency.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter == 0:

            # predict action sequence
            pred_action_seq = self(obs, goal)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.act_seq_len:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:

        self.model.to(dtype=self.dtype)
        self.img_encoder.to(dtype=self.dtype)
        # self.perceiver.to(dtype=self.dtype)
        # self.language_encoder.to(dtype=self.dtype)

        for idx, callback in enumerate(self.trainer.callbacks):
            if isinstance(callback, EMA):
                self.ema_callback_idx = idx
                break

    def forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if self.use_text_not_embedding:
            # latent_goal = self.language_encoder(goal["lang_text"])
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(goal["lang_text"])
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = self.language_encoder(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                obs["rgb_obs"]['rgb_static'].device)

        obs_dict = {
            'rgb_static': einops.rearrange(obs["rgb_obs"]['rgb_static'], 'b t c h w -> (b t) c h w'),
            'rgb_gripper': einops.rearrange(obs["rgb_obs"]['rgb_gripper'], 'b t c h w -> (b t) c h w'),
            'pc_static': einops.rearrange(obs["depth_obs"]['depth_static'], 'b t c h w -> (b t) c h w'),
            'pc_gripper': einops.rearrange(obs["depth_obs"]['depth_gripper'], 'b t c h w -> (b t) c h w')
        }

        obs_dict['latent_goal'] = latent_goal

        perceptual_emb = self.img_encoder(obs_dict)

        act_seq = self.denoise_actions(
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return act_seq

    def on_train_start(self) -> None:

        self.model.to(dtype=self.dtype)
        self.img_encoder.to(dtype=self.dtype)
        # self.perceiver.to(dtype=self.dtype)
        self.language_encoder.to(dtype=torch.float32)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)
