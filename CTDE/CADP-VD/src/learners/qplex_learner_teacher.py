
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam


from components.episode_buffer import EpisodeBatch
from modules.mixers.qplex import DMAQ_QattenMixer
from utils.th_utils import get_params_size



def entropy(x, dim=-1):
    max_entropy = np.log(x.shape[dim])
    x = (x + 1e-8) / torch.sum(x + 1e-8, dim, keepdim=True)

    return (-torch.log(x) * x).sum(dim) / max_entropy



class DMAQ_qattenLearner:

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
            if args.mixer == "dmaq_qatten":
                self.mixer = DMAQ_QattenMixer
            else:
                raise ValueError("Mixer {} not recognized.".format(args.mixer))

            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        if self.args.optimizer == "adam":
            self.optimizer = Adam(params=self.params, lr=args.lr)
        else:
            self.optimizer = RMSprop(
                params=self.params, lr=args.lr, alpha=args.optimi_alpha, eps=args.optim_eps
            )
        

        # a little wasteful to deepcopy (e.g. duplicates act selector), but shoule work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions
    

    def sub_train(
            self,
            batch: EpisodeBatch,
            t_env: int,
            episode_num: int,
            mac,
            mixer,
            optimizer,
            params,
            show_demo=False,
            save_data=None,
    ):
        # Get the relevant quantiles
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]


        # Calculate estimated Q-values
        mac_out = []
        onehot_out = []
        eps = 1e-8
        eye = torch.eye(self.args.n_agents).reshape(-1).to(self.args.device)
        eye = torch.cat([eye] * self.args.att_heads, dim=0)
        self.mac.init_hidden(batch.batch_size)
        att_out = []

        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            att = self.mac.agent.att.dot
            att = att.view(batch.batch_size, -1)
            onehot_out.append(F.kl_div((att + eps).long(), eye, reduction='none').mean(dim=-1))
            mac_out.append(agent_outs)
        
        mac_out = torch.stack(mac_out, dim=1)   # Concat over time
        onehot_out = torch.stack(onehot_out, dim=1)


        # Pick the Q-values for the acts taken by each agent
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
        

        # Calculate the Q-values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        

        # don't need the first timesteps Q-values estimate for calculating targets
        target_mac_out = torch.stack(target_mac_out[1:], dim=1) # Concat across time


        # Mask out unavailable acts.
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999


        # Max over target Q-values
        if self.args.double_q:
            # Get acts that maximize live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = torch.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        
        else:
            # Calculate the Q-values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            
            # don't need the first timesteps Q-value estimate for calculating targets
            target_mac_out = torch.stack(target_mac_out[1:], dim=1) # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]
        

        # Mix
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":

                ans_chosen, \
                q_attend_regs, \
                head_entropies = mixer(
                    chosen_action_qvals, batch["state"][:, :-1], is_v=True
                )

                ans_adv, _, _ = mixer(chosen_action_qvals, 
                                      batch["state"][:, :-1], 
                                      actions=actions_onehot, 
                                      max_q_i=max_action_qvals, 
                                      is_v=False)

                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals, 
                                   batch["state"][:, :-1], 
                                   is_v=False)
                ans_adv = mixer(chosen_action_qvals, 
                                batch["state"][:, :-1], 
                                actions=actions_onehot, 
                                max_q_i=max_action_qvals, 
                                is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            
            
            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals,
                                                            batch["state"][:, 1:],
                                                            is_v=False)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals,
                                                         batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals,
                                                         is_v=False)
                    target_max_qvals = target_chosen + target_adv
                
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals,
                                                      batch["state"][:, 1:],
                                                      is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals,
                                                   batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals,
                                                   is_v=False)
                    target_max_qvals = target_chosen + target_adv
                
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)
        

        # Calculate 1-step Q-learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print(
                'action_pair_%d_%d' % (save_data[0], save_data[1]),
                np.squeeze(q_data[:, 0]),
                np.squeeze(q_i_data[:, 0]),
                np.squeeze(tot_q_data[:, 0]),
                np.squeeze(tot_target[:, 0])
            )
            self.logger.log_stat(
                'action_pair_%d_%d' % (save_data[0], save_data[1]),
                np.squeeze(tot_q_data[:, 0]), t_env
            )

            return
        

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)


        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask


        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
        
        masked_hit_prob = torch.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()


        # Optimize
        if self.args.learner == "qplex_learner" and t_env > self.args.breakpoint:
            onehot_out = onehot_out[:, :-1].unsqueeze(-1) * mask
            loss = loss + self.args.alpha * onehot_out.sum() / mask.sum()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimizer.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("train/loss", loss.item(), t_env)
            self.logger.log_stat("train/hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("train/grad_norm", grad_norm.item(), t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("train/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("train/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("train/target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
    

    def train(
            self,
            batch: EpisodeBatch,
            t_env: int,
            episode_num: int,
            show_demo=False,
            save_data=None,
    ):
        self.sub_train(batch,
                       t_env,
                       episode_num,
                       self.mac,
                       self.mixer,
                       self.optimizer,
                       self.params,
                       show_demo=show_demo,
                       save_data=save_data)
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
    

    