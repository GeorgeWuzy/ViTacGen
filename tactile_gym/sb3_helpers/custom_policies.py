# create a class clalled RAD_SAC that inherits from SAC
# add augmentations and visualise_aug to the __init__ function
from typing import Optional


from sb3_contrib.rad.rad_ppo import RAD_PPO
from stable_baselines3.common.preprocessing import preprocess_obs

from sb3_contrib import RAD_SAC

class CURL_SAC(RAD_SAC):
    def __init__(
            self,
            *args,
            curl_alpha: float = 0.1,  # Coefficient for CURL loss
            curl_tau: float = 0.05,  # Soft update coefficient for CURL
            **kwargs
    ):
        super(CURL_SAC, self).__init__(*args, **kwargs)
        self.curl_alpha = curl_alpha
        self.curl_tau = curl_tau

    def train(self, gradient_steps: int, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train(gradient_steps, batch_size)

        # Train CURL
        if self.augmentations is not None:
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = replay_data.observations

            # with th.no_grad():
            #     target_obs = self.replay_buffer.augment_obs(obs.clone()).detach()
            # preprocess the observations
            preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=True)
            # compute the infoNCE loss
            infoNCE_loss = self.policy.actor.features_extractor.compute_loss(preprocessed_obs, self.curl_tau)

            # Backpropagate CURL loss and update the online encoder
            self.policy.actor.optimizer.zero_grad()
            infoNCE_loss.backward()
            self.policy.actor.optimizer.step()

            # Soft update of the momentum encoder
            self.actor.features_extractor.update_momentum_encoder(self.curl_tau)

            self.logger.record("train/infoNCE_loss", infoNCE_loss.item())


class MViTacRL_SAC(RAD_SAC):
    def __init__(
            self,
            *args,
            beta: float = 0.1,  # Coefficient for CURL loss
            tau: float = 0.05,  # Soft update coefficient for CURL
            lambda_vis: float = 1,
            lambda_tac: float = 1,
            lambda_vis_tac: float = 1,
            # augmentations: Dict[str, th.nn.Sequential] = None,
            **kwargs
    ):
        # self.augmentations = augmentations
        # self.augmentations = augmentations["visual"]
        # self.augmentations_tactile = augmentations["tactile"]
        # kwargs["augmentations"] = self.augmentations
        super(MViTacRL_SAC, self).__init__(*args, **kwargs)
        self.beta = beta
        self.tau = tau
        self.lambda_vis = lambda_vis
        self.lambda_tac = lambda_tac
        self.lambda_vis_tac = lambda_vis_tac

    def train(self, gradient_steps: int, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train(gradient_steps, batch_size)

        # Train CURL
        if self.augmentations is not None:
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = replay_data.observations
            # preprocess the observations
            obs = preprocess_obs(obs, self.observation_space, normalize_images=True)

            # get the tactile observations and convert to dict
            tactile_obs = obs['tactile']
            # get the visual observations
            visual_obs = obs['visual']

            # compute the infoNCE loss
            losses = self.policy.actor.features_extractor.compute_loss(visual_obs, tactile_obs)
            infoNCE_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter = losses

            # weighted sum of the losses using the lambda parameters
            infoNCE_loss = (self.lambda_vis * vis_loss_intra
                            + self.lambda_tac * tac_loss_intra
                            + self.lambda_vis_tac * vis_tac_inter + self.lambda_vis_tac * tac_vis_inter)

            # weight the losses
            vis_loss_intra = vis_loss_intra * self.beta

            # Backpropagate CURL loss and update the online encoder
            self.policy.actor.optimizer.zero_grad()
            infoNCE_loss.backward()
            self.policy.actor.optimizer.step()

            try:
                # Soft update of the momentum encoder
                self.actor.features_extractor.momentum_update_key_encoder()
            except Exception as e:
                pass

            self.logger.record("train/infoNCE_loss", infoNCE_loss.item())
            self.logger.record("train/vis_loss_intra", vis_loss_intra.item())
            self.logger.record("train/tac_loss_intra", tac_loss_intra.item())
            self.logger.record("train/vis_tac_inter", vis_tac_inter.item())
            self.logger.record("train/tac_vis_inter", tac_vis_inter.item())

class MViTacRL_PPO(RAD_PPO):
    def __init__(
            self,
            *args,
            beta: float = 0.1,  # Coefficient for CURL loss
            tau: float = 0.05,  # Soft update coefficient for CURL
            lambda_vis: float = 1,
            lambda_tac: float = 1,
            lambda_vis_tac: float = 1,
            # augmentations: Dict[str, th.nn.Sequential] = None,
            **kwargs
    ):
        # self.augmentations = augmentations
        # self.augmentations = augmentations["visual"]
        # self.augmentations_tactile = augmentations["tactile"]
        # kwargs["augmentations"] = self.augmentations
        super(MViTacRL_PPO, self).__init__(*args, **kwargs)
        self.beta = beta
        self.tau = tau
        self.lambda_vis = lambda_vis
        self.lambda_tac = lambda_tac
        self.lambda_vis_tac = lambda_vis_tac

    # def _setup_model(self) -> None:
    #     super(RAD_SAC, self)._setup_model()
    #
    #     try:
    #         n_stack = self.env.n_stack
    #     except:
    #         n_stack = 1
    #
    #     self.replay_buffer = MMAugmentedDictReplayBuffer(
    #         self.buffer_size,
    #         self.observation_space,
    #         self.action_space,
    #         self.device,
    #         augmentations=self.augmentations,
    #         augmentations_tactile=self.augmentations_tactile,
    #         visualise_aug=self.visualise_aug,
    #         n_stack=n_stack,
    #         n_envs=self.n_envs,
    #         optimize_memory_usage=self.optimize_memory_usage,
    #     )
    def train(self, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train()

        # Train CURL
        if self.augmentations is not None:
            # Sample from the rollout buffer
            # buffer_size = len(self.rollout_buffer.returns)  # Assuming `returns` attribute stores the buffer size
            # batch_inds = np.random.randint(0, buffer_size, size=batch_size)  # Random indices for sampling
            # replay_data = self.rollout_buffer._get_samples(batch_inds)
            for _ in range(self.n_epochs):
                replay_data = self.rollout_buffer.sample(self.batch_size, env=self._vec_normalize_env)
                obs = replay_data.observations
                # preprocess the observations
                obs = preprocess_obs(obs, self.observation_space, normalize_images=True)

                # get the tactile observations and convert to dict
                tactile_obs = obs['tactile']
                # get the visual observations
                visual_obs = obs['visual']

                # compute the infoNCE loss
                losses = self.policy.features_extractor.compute_loss(visual_obs, tactile_obs)
                infoNCE_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter = losses

                # weighted sum of the losses using the lambda parameters
                infoNCE_loss = (self.lambda_vis * vis_loss_intra
                                + self.lambda_tac * tac_loss_intra
                                + self.lambda_vis_tac * vis_tac_inter + self.lambda_vis_tac * tac_vis_inter)

                # weight the losses
                vis_loss_intra = vis_loss_intra * self.beta

                # Backpropagate CURL loss and update the online encoder
                self.policy.optimizer.zero_grad()
                infoNCE_loss.backward()
                self.policy.optimizer.step()

                # Soft update of the momentum encoder
                self.policy.features_extractor.momentum_update_key_encoder()

                self.logger.record("train/infoNCE_loss", infoNCE_loss.item())
                self.logger.record("train/vis_loss_intra", vis_loss_intra.item())
                self.logger.record("train/tac_loss_intra", tac_loss_intra.item())
                self.logger.record("train/vis_tac_inter", vis_tac_inter.item())
                self.logger.record("train/tac_vis_inter", tac_vis_inter.item())

class CL_PPO(RAD_PPO):
    def init(self,
            *args,
            **kwargs,
    ):
        super(CL_PPO, self).__init__(*args, **kwargs)
        self.cl_loss = False

    def set_param(self, cl_loss=False):
        self.cl_loss = cl_loss
        print("CL loss:", cl_loss)
    
    def train(self, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train()
        if self.cl_loss:
            for _ in range(10):
                replay_data = self.rollout_buffer.sample(self.batch_size, env=self._vec_normalize_env)
                obs = replay_data.observations
                # preprocess the observations
                obs = preprocess_obs(obs, self.observation_space, normalize_images=True)

                # get the tactile observations and convert to dict
                tactile_obs = obs['tactile']
                # get the visual observations
                visual_obs = obs['visual']

                # compute the infoNCE loss
                loss = self.policy.features_extractor.compute_loss(visual_obs, tactile_obs)

                # Backpropagate CURL loss and update the online encoder
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

                # Soft update of the momentum encoder
                self.policy.features_extractor.momentum_update_key_encoder()

class CL_SAC(RAD_SAC):
    def init(self,
            *args,
            **kwargs,
    ):
        super(CL_SAC, self).__init__(*args, **kwargs)
        self.cl_loss = False

    def set_param(self, cl_loss=False):
        self.cl_loss = cl_loss
        print("CL loss:", cl_loss)
    
    def train(self, gradient_steps: int, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train(gradient_steps, batch_size)
        if self.cl_loss:
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = replay_data.observations
            # preprocess the observations
            obs = preprocess_obs(obs, self.observation_space, normalize_images=True)

            # get the tactile observations and convert to dict
            tactile_obs = obs['tactile']
            # get the visual observations
            visual_obs = obs['visual']

            # compute the infoNCE loss
            loss = self.policy.actor.features_extractor.compute_loss(visual_obs, tactile_obs)

            # weight the losses
            vis_loss_intra = vis_loss_intra * self.beta

            # Backpropagate CURL loss and update the online encoder
            self.policy.actor.optimizer.zero_grad()
            loss.backward()
            self.policy.actor.optimizer.step()

            # Soft update of the momentum encoder
            self.actor.features_extractor.momentum_update_key_encoder()