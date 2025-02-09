
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner



def _t2n(x):

    return x.detach().cpu().numpy()



class SMACRunner(Runner):
    """
    Runner class to perform training, eval. and data collection for SMAC.
    See parent class for details.
    """

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
    

    def run(self):
        self.warmup()


        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length \
                // self.n_rollout_threads
        
        tmp_timestep = 0

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)


        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            for step in range(self.episode_length):
                # Sample acts
                values, actions, action_log_probs, \
                rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, \
                infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, \
                    available_actions, values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic
                

                # Insert data into buffer.
                self.insert(data)
            

            # Compute return and update network.
            self.compute()
            train_infos = self.train()


            # Post process.
            total_num_steps = (episode + 1) * self.episode_length \
                            * self.n_rollout_threads
            
            # Save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
            

            if getattr(self.all_args, 'use_CADP', False):
                if (total_num_steps - tmp_timestep) - 500000 > 0:
                    tmp_timestep = total_num_steps
                    self.save_timestep(total_num_steps)
                
                if total_num_steps > self.all_args.cadp_breakpoint:
                    self.trainer.use_cadp_loss = True
            

            # log info.
            if episode % self.log_interval == 0:
                end = time.time()

                print(
                    "\n Map {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, EPS {}.\n"
                    .format(
                        self.all_args.map_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []


                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(
                                info[0]['battles_won'] - last_battles_won[i]
                            )
                        
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['baattles_game'])
                            incre_battles_game.append(
                                info[0]['battles_game'] - last_battles_game[i]
                            )
                    
                    incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) \
                                    if np.sum(incre_battles_game) > 0 else 0.0
                    
                    print("incre win rate is {}".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log(
                            {"incre_win_rate": incre_win_rate},
                            step=total_num_steps,
                        )

                    else:
                        self.writter.add_scalars(
                            "incre_win_rate",
                            {"incre_win_rate": incre_win_rate},
                            total_num_steps,
                        )
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                
                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() \
                    / reduce(lambda x, y: x * y, list(self.buffer.active_masks.shape))
                
                self.log_train(train_infos, total_num_steps)
            
            
            # Eval.
            if episode % self.eval_interval == 0 and self.use_eval:
                if getattr(self.all_args, 'use_CADP', False):

                    self.policy.actor.use_att_v = True
                    self.eval(total_num_steps, "student_")

                    self.policy.actor.use_att_v = False
                    self.eval(total_num_steps, "teacher_")

                    pass

                else:
                    self.eval(total_num_steps)
    

    def warmup(self):
        
        # reset env.
        obs, share_obs, available_actions = self.envs.reset()


        # replay buffer.
        if not self.use_centralized_V:
            share_obs = obs
        

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()
    

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        value, action, action_log_prob, \
        rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )

        
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))


        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    

    