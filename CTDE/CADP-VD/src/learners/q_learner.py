
import copy
import torch
from torch.optim import RMSprop


from components.episode_buffer import EpisodeBatch
from modules.miexers.vdn import VDNMixer
from modules.mixers.qmix import QMixer




class QLearner:

    def __init__(
            self,
            mac,
            scheme,
            logger,
            args
    ):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            
            elif self.mixer == "qmix":
                self.mixer = QMixer(args)
            
            else:
                raise ValueError("Mixer {} not recongnised.".format(args.mixer))

            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        self.optimizer = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        # a little wasteful to deepcopy (e.g. duplicates act selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
    

    def train(
            self,
            batch: EpisodeBatch,
            t_env: int,
            episode_num: int
    ):
        # Get the relevant quantiles
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]


        # Calculate estimated Q-values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        
        mac_out = torch.stack(mac_out, dim=1)   # Concat over time.


        # Pick the Q-values for the acts taken by each agent
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)


        # Calculate the Q-values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)


        # don't need the first timesteps Q-value estimate for calculating targets
        target_mac_out = torch.stack(target_mac_out[1:], dim=1) # Concat across time


        # Mask out unavailable acts
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999


        # Max over target Q-values
        if self.args.double_q:
            # Get acts that maximize live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_mac_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_mac_actions).squeeze(3)
        
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])


        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals


        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)


        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask


        # Normal L2 loss, taken mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()


        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
    

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        self.logger.console_logger.info("Updated target network")
    

    