import time

import torch
import os
import json
import numpy as np
from coin_game_envs import CoinGamePPO, SymmetricCoinGame
from coin_game_mfos_agent import MemoryMFOS, PPOMFOS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="milad_self")
parser.add_argument("--grid-size", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":
    # seed torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    ############## Hyperparameters ##############
    batch_size = 8  # 8192 #, 32768
    state_dim = [7, args.grid_size, args.grid_size]
    action_dim = 4
    n_latent_var = 16  # number of variables in hidden layer

    # traj_length = 32

    max_episodes = 1000  # max training episodes
    log_interval = 50

    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.995  # discount factor
    tau = 0.3  # GAE

    traj_length = 16

    save_freq = 50
    K_epochs = 16  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    use_gae = False

    inner_ep_len = 16
    num_steps = 256  # , 500

    lamb = 1.0
    lamb_anneal = 0.005
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    memory_0 = MemoryMFOS()
    ppo_0 = PPOMFOS(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, inner_ep_len)

    memory_1 = MemoryMFOS()
    ppo_1 = PPOMFOS(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, inner_ep_len)

    print(lr, betas)
    print(sum(p.numel() for p in ppo_0.policy_old.parameters() if p.requires_grad))

    env = SymmetricCoinGame(batch_size, inner_ep_len, grid_size=args.grid_size, coin_game_type='lola')
    # env
    nl_env = CoinGamePPO(batch_size, inner_ep_len, grid_size=args.grid_size, coin_game_type='lola')

    start_time = time.time()
    # training loop
    for i_episode in range(1, max_episodes + 1):
        print("=" * 10)
        print(f"episode: {i_episode}")
        print(f"lamb: {lamb}")

        if lamb > 0:
            lamb -= lamb_anneal


        def evaluate_agent_0_against_always_defect():
            state_0, state_1 = env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_reward_1 = torch.zeros(batch_size).to(device)

            p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0
            for t in range(num_steps):
                # Running policy_old:
                if t % inner_ep_len == 0:
                    ppo_0.policy_old.reset(memory_0, t == 0)

                with torch.no_grad():
                    action_0 = ppo_0.policy_old.act(state_0.detach())
                    action_1 = env.env.get_moves_shortest_path_to_coin(red_agent_perspective=False)
                states, rewards, info_2 = env.step([action_0, action_1])
                state_0, state_1 = states
                reward_0, reward_1 = rewards

                running_reward_0 += reward_0.squeeze(-1)
                running_reward_1 += reward_1.squeeze(-1)

                memory_0.rewards.append(reward_0.detach())

                if info_2 is not None:
                    p1_num_opp += info_2[1]
                    p2_num_opp += info_2[2]
                    p2_num_self += info_2[3]
                    p1_num_self += info_2[0]

            ppo_0.policy_old.reset(memory_0)

            memory_0.clear_memory()
            return {
                'ad_reward_agent_0_against_always_defect': running_reward_0.mean().item(),
                'ad_reward_always_defect_against_agent0': running_reward_1.mean().item(),
                'ad_agent0_takes_always_defect_coin': p1_num_opp.float().mean().item(),
                'ad_agent0_takes_agent0_coin': p1_num_self.float().mean().item(),
                'ad_always_defect_takes_agent0_coin': p2_num_opp.float().mean().item(),
                'ad_always_defect_takes_always_defect_coin': p2_num_self.float().mean().item(),
            }


        if np.random.random() > lamb:
            print("v opponent")
            state_0, state_1 = env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_reward_1 = torch.zeros(batch_size).to(device)
            p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0
            for t in range(num_steps):
                # Running policy_old:
                if t % inner_ep_len == 0:
                    ppo_0.policy_old.reset(memory_0, t == 0)
                    ppo_1.policy_old.reset(memory_1, t == 0)

                with torch.no_grad():
                    action_0 = ppo_0.policy_old.act(state_0.detach())
                    action_1 = ppo_1.policy_old.act(state_1.detach())
                states, rewards, info_2 = env.step([action_0, action_1])
                state_0, state_1 = states
                reward_0, reward_1 = rewards

                running_reward_0 += reward_0.squeeze(-1)
                running_reward_1 += reward_1.squeeze(-1)

                memory_0.rewards.append(reward_0.detach())
                memory_1.rewards.append(reward_1.detach())

                if info_2 is not None:
                    p1_num_opp += info_2[1]
                    p2_num_opp += info_2[2]
                    p2_num_self += info_2[3]
                    p1_num_self += info_2[0]

            ppo_0.policy_old.reset(memory_0)
            ppo_1.policy_old.reset(memory_1)

            ppo_0.update(memory_0)
            ppo_1.update(memory_1)

            memory_0.clear_memory()
            memory_1.clear_memory()

            stats = {
                "ep": i_episode,
                "other": True,
                "reward 0": running_reward_0.mean().item(),
                "reward 1": running_reward_1.mean().item(),
                "p1_opp": p1_num_opp.float().mean().item(),
                "p2_opp": p2_num_opp.float().mean().item(),
                "p1_self": p1_num_self.float().mean().item(),
                "p2_self": p2_num_self.float().mean().item(),
            }
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            state = nl_env.reset()
            running_reward_0 = torch.zeros(batch_size).to(device)
            opp_running_reward_0 = torch.zeros(batch_size).to(device)
            p1_num_opp_0, p2_num_opp_0, p1_num_self_0, p2_num_self_0 = 0, 0, 0, 0
            for t in range(num_steps):
                # Running policy_old:
                if t % inner_ep_len == 0:
                    ppo_0.policy_old.reset(memory_0, t == 0)
                with torch.no_grad():
                    action = ppo_0.policy_old.act(state.detach())
                state, reward, done, info, info_2 = nl_env.step(action.detach())
                running_reward_0 += reward.detach()
                opp_running_reward_0 += info.detach()
                memory_0.rewards.append(reward.detach())
                if info_2 is not None:
                    p1_num_opp_0 += info_2[2]
                    p2_num_opp_0 += info_2[1]
                    p1_num_self_0 += info_2[3]
                    p2_num_self_0 += info_2[0]

            ppo_0.policy_old.reset(memory_0)
            ppo_0.update(memory_0)
            memory_0.clear_memory()

            state = nl_env.reset()
            running_reward_1 = torch.zeros(batch_size).to(device)
            opp_running_reward_1 = torch.zeros(batch_size).to(device)
            p1_num_opp_1, p2_num_opp_1, p1_num_self_1, p2_num_self_1 = 0, 0, 0, 0
            for t in range(num_steps):
                # Running policy_old:
                if t % inner_ep_len == 0:
                    ppo_1.policy_old.reset(memory_1, t == 0)
                with torch.no_grad():
                    action = ppo_1.policy_old.act(state.detach())
                state, reward, done, info, info_2 = nl_env.step(action.detach())
                running_reward_1 += reward.detach()
                opp_running_reward_1 += info.detach()
                memory_1.rewards.append(reward.detach())
                if info_2 is not None:
                    p1_num_opp_1 += info_2[2]
                    p2_num_opp_1 += info_2[1]
                    p1_num_self_1 += info_2[3]
                    p2_num_self_1 += info_2[0]

            ppo_1.policy_old.reset(memory_1)
            ppo_1.update(memory_1)
            memory_1.clear_memory()
            stats = {
                "ep": i_episode,
                "other": False,
                "reward 0": running_reward_0.mean().item(),
                "reward 1": running_reward_1.mean().item(),
                "opp reward 0": opp_running_reward_0.mean().item(),
                "opp reward 1": opp_running_reward_1.mean().item(),
                "p1_num_opp_0": p1_num_opp_0.float().mean().item(),
                "p2_num_opp_0": p2_num_opp_0.float().mean().item(),
                "p1_num_self_0": p1_num_self_0.float().mean().item(),
                "p2_num_self_0": p2_num_self_0.float().mean().item(),
                "p1_num_opp_1": p1_num_opp_1.float().mean().item(),
                "p2_num_opp_1": p2_num_opp_1.float().mean().item(),
                "p1_num_self_1": p1_num_self_1.float().mean().item(),
                "p2_num_self_1": p2_num_self_1.float().mean().item(),
            }
        stats.update({"walltime": time.time() - start_time})
        if i_episode % save_freq == 0:
            stats.update(evaluate_agent_0_against_always_defect())
            stats.update({"walltime": time.time() - start_time})
            ppo_0.save(os.path.join(name, f"{i_episode}_0.pth"))
            ppo_1.save(os.path.join(name, f"{i_episode}_1.pth"))
            with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
                json.dump(stats, f)
            print(f"SAVING! {i_episode}")

        print(f"stats: {stats}")
