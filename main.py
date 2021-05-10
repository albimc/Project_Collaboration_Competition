# General Libraries
import os
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

# PyTorch
import torch
# Environment
from unityagents import UnityEnvironment
# Replay Buffer
from replaybuffer import ReplayBuffer
# Multi-Agent
from maddpg import MADDPG
# Utilities
from utilities import seeding

# Check Progress
from tensorboardX import SummaryWriter
import progressbar as pb
import time

# Debug
# import pdb
# pdb.set_trace()


def main():

    ##########
    # CONFIG #
    ##########
    # Target Reward
    tgt_score = 0.5
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed
    seed = 7
    seeding(seed)
    # Model Architecture
    # Actor
    hidden_in_actor = 256
    hidden_out_actor = 128
    lr_actor = 1e-4
    # Critic
    hidden_in_critic = 256
    hidden_out_critic = 128
    lr_critic = 3e-4
    weight_decay_critic = 0
    # Episodes
    number_of_episodes = 10000
    episode_length = 2000
    # Buffer
    buffer_size = int(1e6)
    batchsize = 512
    # Agent Update Frequency
    episode_per_update = 1
    # Rewards Discounts Factor
    discount_factor = 0.95
    # Soft Update Weight
    tau = 1e-2
    # Noise Process
    noise_factor = 2
    noise_reduction = 0.9999
    noise_floor = 0.0
    # Window
    win_len = 100
    # Save Frequency
    save_interval = 200
    # Logger
    log_path = os.getcwd()+"/log"
    logger = SummaryWriter(log_dir=log_path)
    # Model Directory
    model_dir = os.getcwd()+"/model_dir"
    os.makedirs(model_dir, exist_ok=True)
    # Load Saved Model
    load_model = False

    ####################
    # Load Environment #
    ####################
    env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")
    # Get brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print('Brain Name:', brain_name)
    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # Number of Agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    ####################
    # Show Progressbar #
    ####################
    widget = ['episode: ', pb.Counter(), '/', str(number_of_episodes), ' ',
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ']
    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()
    start = time.time()

    ###############
    # Multi Agent #
    ###############
    maddpg = MADDPG(state_size, action_size, num_agents,
                    hidden_in_actor, hidden_out_actor, lr_actor,
                    hidden_in_critic, hidden_out_critic, lr_critic, weight_decay_critic,
                    discount_factor, tau, seed, device)

    if load_model:
        load_dict_list = torch.load(os.path.join(model_dir, 'episode-saved.pt'))
        for i in range(num_agents):
            maddpg.maddpg_agent[i].actor.load_state_dict(load_dict_list[i]['actor_params'])
            maddpg.maddpg_agent[i].actor_optimizer.load_state_dict(load_dict_list[i]['actor_optim_params'])
            maddpg.maddpg_agent[i].critic.load_state_dict(load_dict_list[i]['critic_params'])
            maddpg.maddpg_agent[i].critic_optimizer.load_state_dict(load_dict_list[i]['critic_optim_params'])

    #################
    # Replay Buffer #
    #################
    rebuffer = ReplayBuffer(buffer_size, seed, device)

    #################
    # TRAINING LOOP #
    #################
    # initialize scores
    scores_history = []
    scores_window = deque(maxlen=save_interval)

    # i_episode = 0
    for i_episode in range(number_of_episodes):
        timer.update(i_episode)

        # Reset Environmet
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)

        # Reset Agent
        maddpg.reset()

        # episode_t = 0
        for episode_t in range(episode_length):

            # Explore with decaying noise factor
            actions = maddpg.act(states, noise_factor=noise_factor)
            env_info = env.step(actions)[brain_name]             # Environment reacts
            next_states = env_info.vector_observations           # get the next states
            rewards = env_info.rewards                           # get the rewards
            dones = env_info.local_done                          # see if episode has finished

            ###################
            # Save Experience #
            ###################
            rebuffer.add(states, actions, rewards, next_states, dones)

            scores += rewards
            states = next_states

            if any(dones):
                break

        scores_history.append(np.max(scores))       # save most recent score
        scores_window.append(np.max(scores))        # save most recent score
        avg_rewards = np.mean(scores_window)
        noise_factor = max(noise_floor, noise_factor*noise_reduction)    # Reduce Noise Factor

        #########
        # LEARN #
        #########
        if len(rebuffer) > batchsize and i_episode % episode_per_update == 0:
            for a_i in range(num_agents):
                samples = rebuffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            # Soft Update
            maddpg.update_targets()

        ##################
        # Track Progress #
        ##################
        if i_episode % save_interval == 0 or i_episode == number_of_episodes-1:
            logger.add_scalars('rewards', {'Avg Reward': avg_rewards, 'Noise Factor': noise_factor}, i_episode)
            print('\nElapsed time {:.1f} \t Update Count {} \t Last Episode t {}'.format((time.time() - start)/60, maddpg.update_count, episode_t),
                  '\nEpisode {} \tAverage Score: {:.2f} \tNoise Factor {:2f}'.format(i_episode, avg_rewards, noise_factor), end="\n")

        ##############
        # Save Model #
        ##############
        save_info = ((i_episode) % save_interval == 0 or i_episode == number_of_episodes)
        if save_info:
            save_dict_list = []
            for i in range(num_agents):
                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)
            torch.save(save_dict_list, os.path.join(model_dir, 'episode-Latest.pt'))

            pd.Series(scores_history).to_csv(os.path.join(model_dir, "scores.csv"))

            # plot the scores
            rolling_mean = pd.Series(scores_history).rolling(win_len).mean()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores_history)), scores_history)
            plt.axhline(y=tgt_score, color='r', linestyle='dashed')
            plt.plot(rolling_mean, lw=3)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            # plt.show()
            fig.savefig(os.path.join(model_dir, 'Average_Score.pdf'))
            fig.savefig(os.path.join(model_dir, 'Average_Score.jpg'))
            plt.close()

        if avg_rewards > tgt_score:
            logger.add_scalars('rewards', {'Avg Reward': avg_rewards, 'Noise Factor': noise_factor}, i_episode)
            print('\nElapsed time {:.1f} \t Update Count {} \t Last Episode t {}'.format((time.time() - start)/60, maddpg.update_count, episode_t),
                  '\nEpisode {} \tAverage Score: {:.2f} \tNoise Factor {:2f}'.format(i_episode, avg_rewards, noise_factor), end="\n")
            break

    env.close()
    logger.close()
    timer.finish()


if __name__ == '__main__':
    main()
