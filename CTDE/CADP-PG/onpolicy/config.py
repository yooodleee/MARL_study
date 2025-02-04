
import argparse



def get_config():
    """
    The configuration parser for common hyperparameters of all env.
    Please reach `scripts/train/<env>_runner.py` file to find private hyperparameters
        only used in <env>.

    Prepare parameters
    ---------------------
        --algorithm_name <algorithm_name>
            specify the algo, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`.
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some envs.
        --num_env_steps <int>
            number of env steps to train (defaultL: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --user_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use
                tensorboard to log data.

    Env params
    -------------
        --env_name <str>
            specify the name of env
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will
                use concatenated local obs.

                
    Replay Buffer params
    ------------------------
        --episode_length <int>
            the max length of episode in the buffer.

            
    Network params
    -------------------
        --share_policy
            by default True, all agents will share the same network; set to make training
                agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized
                training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks.
        --layer_N <int>
            Number of layers for actor/critic networks.
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0
                initialization for biases.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_navie_recurrent_policy
            by default False, use the whole trajecotry to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1 ).
        --data_check_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.


    Optimizer params
    -------------------
        --lr <float>
            learning rate parameter, (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic (default: 5e-4, fixed).
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)


    PPO params
    ---------------
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, 
                do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. 
            If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.


    PPG params
    -----------------------
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
            
    Run params
    ----------------------------
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. 
            If set, use a linear schedule on the learning rate
    
            
    Save & Log params
    ------------------------------
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
            
    Eval params
    -----------------
        --use_eval
            by default, do not start evaluation. If set`, start evaluation 
                alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

            
    Render params
    -------------------
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, 
                start render. Note: something, the environment has internal 
                render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
        
    Pretrained params
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    