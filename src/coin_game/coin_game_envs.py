from random import random

import torch
import numpy as np
from coin_game_ppo_agent import PPO, Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CoinGameGPU:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = torch.stack(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([0, -1]),
            torch.LongTensor([1, 0]),
            torch.LongTensor([-1, 0]),
        ],
        dim=0,
    ).to(device)

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None

    def reset(self):
        self.step_count = 0

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)

        self.red_coin_pos = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1)
        self.blue_coin_pos = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1)

        state = self._generate_state()
        observations = [state, state]
        return observations

    def _generate_coins(self):
        mask_red = torch.logical_or(self._same_pos(self.red_coin_pos, self.blue_pos), self._same_pos(self.red_coin_pos, self.red_pos))
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)[mask_red]
        self.red_coin_pos[mask_red] = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1)

        mask_blue = torch.logical_or(self._same_pos(self.blue_coin_pos, self.blue_pos), self._same_pos(self.blue_coin_pos, self.red_pos))
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)[mask_blue]
        self.blue_coin_pos[mask_blue] = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        red_coin_pos_flat = self.red_coin_pos[:, 0] * self.grid_size + self.red_coin_pos[:, 1]
        blue_coin_pos_flat = self.blue_coin_pos[:, 0] * self.grid_size + self.blue_coin_pos[:, 1]

        state = torch.zeros((self.batch_size, 4, self.grid_size * self.grid_size)).to(device)

        state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
        state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)
        state[:, 2].scatter_(1, red_coin_pos_flat[:, None], 1)
        state[:, 3].scatter_(1, blue_coin_pos_flat[:, None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_reward = torch.zeros(self.batch_size).to(device)
        red_red_matches = self._same_pos(self.red_pos, self.red_coin_pos)
        red_reward[red_red_matches] += 1
        red_blue_matches = self._same_pos(self.red_pos, self.blue_coin_pos)
        red_reward[red_blue_matches] += 1

        blue_reward = torch.zeros(self.batch_size).to(device)
        blue_red_matches = self._same_pos(self.blue_pos, self.red_coin_pos)
        blue_reward[blue_red_matches] += 1
        blue_blue_matches = self._same_pos(self.blue_pos, self.blue_coin_pos)
        blue_reward[blue_blue_matches] += 1

        red_reward[blue_red_matches] -= 2
        blue_reward[red_blue_matches] -= 2

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        observations = [state, state]
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(device)
        else:
            done = torch.zeros(self.batch_size).to(device)

        return observations, reward, done, (red_red_matches.sum(), red_blue_matches.sum(), blue_red_matches.sum(), blue_blue_matches.sum())

class LOLACoinGameGPU: # just one coin at a time, like LOLA, POLA, and LOQA. Also, flips the coin color when it is picked up, not randomly.
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = torch.stack(
        [
            torch.LongTensor([0, 1]), # right
            torch.LongTensor([0, -1]), # left
            torch.LongTensor([1, 0]), # down
            torch.LongTensor([-1, 0]), # up
        ],
        dim=0,
    ).to(device)

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None

    def reset(self):
        self.step_count = 0

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        coin_pos_flat = self.generate_random_pos_for_coins_that_are_not_on_agent_pos()
        self.coin_pos = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)
        self.coin_color = torch.randint(2, size=(self.batch_size,)).to(device)

        state = self._generate_state()
        observations = [state, state]
        return observations

    def generate_random_pos_for_coins_that_are_not_on_agent_pos(self):
        mask_flat = torch.zeros((self.batch_size, self.grid_size * self.grid_size)).to(device)
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]
        mask_flat = mask_flat.scatter_(1, red_pos_flat[:, None], 1)
        mask_flat = mask_flat.scatter_(1, blue_pos_flat[:, None], 1)
        unoccupied_pos_flat = 1 - mask_flat
        random_pos = []
        for i in range(self.batch_size):
            unoccupied_indices = unoccupied_pos_flat[i].nonzero().view(-1).tolist()
            choice = torch.randint(0, len(unoccupied_indices), (1,))
            idx = unoccupied_indices[choice]
            random_pos.append(idx)
        random_pos = torch.tensor(random_pos).to(device)
        return random_pos

    def _generate_coins(self):
        mask = torch.logical_or(self._same_pos(self.coin_pos, self.blue_pos), self._same_pos(self.coin_pos, self.red_pos))
        coin_pos_flat = self.generate_random_pos_for_coins_that_are_not_on_agent_pos()[mask]
        self.coin_pos[mask] = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)
        self.coin_color[mask] = 1 - self.coin_color[mask] # flip coin color as in POLA and LOLA and LOQA

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        coin_pos_flat = self.coin_pos[:, 0] * self.grid_size + self.coin_pos[:, 1]

        state = torch.zeros((self.batch_size, 4, self.grid_size * self.grid_size)).to(device)

        state[:, 0] = state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
        state[:, 1] = state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)

        red_mask = self.coin_color == 0
        state[red_mask, 2] = state[red_mask, 2].scatter_(1, coin_pos_flat[red_mask, None], 1)
        blue_mask = self.coin_color == 1
        state[blue_mask, 3] = state[blue_mask, 3].scatter_(1, coin_pos_flat[blue_mask, None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_reward = torch.zeros(self.batch_size).to(device)
        red_red_matches = self._same_pos(self.red_pos, self.coin_pos) & (self.coin_color == 0)
        red_reward[red_red_matches] += 1
        red_blue_matches = self._same_pos(self.red_pos, self.coin_pos) & (self.coin_color == 1)
        red_reward[red_blue_matches] += 1

        blue_reward = torch.zeros(self.batch_size).to(device)
        blue_red_matches = self._same_pos(self.blue_pos, self.coin_pos) & (self.coin_color == 0)
        blue_reward[blue_red_matches] += 1
        blue_blue_matches = self._same_pos(self.blue_pos, self.coin_pos) & (self.coin_color == 1)
        blue_reward[blue_blue_matches] += 1

        red_reward[blue_red_matches] -= 2
        blue_reward[red_blue_matches] -= 2

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        observations = [state, state]
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(device)
        else:
            done = torch.zeros(self.batch_size).to(device)

        return observations, reward, done, (red_red_matches.sum(), red_blue_matches.sum(), blue_red_matches.sum(), blue_blue_matches.sum())

    def get_moves_shortest_path_to_coin(self, red_agent_perspective=True):
        # Ties broken arbitrarily, in this case, since I check the vertical distance later
        # priority is given to closing vertical distance (making up or down moves)
        # before horizontal moves
        if red_agent_perspective:
            agent_pos = self.red_pos
        else:
            agent_pos = self.blue_pos
        actions = torch.zeros(self.batch_size) - 1
        # assumes red agent perspective
        horiz_dist_right = (self.coin_pos[:, 1] - agent_pos[:, 1]) % self.grid_size
        horiz_dist_left = (agent_pos[:, 1] - self.coin_pos[:, 1]) % self.grid_size

        vert_dist_down = (self.coin_pos[:, 0] - agent_pos[:,
                                                0]) % self.grid_size
        vert_dist_up = (agent_pos[:, 0] - self.coin_pos[:,
                                          0]) % self.grid_size
        actions[horiz_dist_right < horiz_dist_left] = 0
        actions[horiz_dist_left < horiz_dist_right] = 1
        actions[vert_dist_down < vert_dist_up] = 2
        actions[vert_dist_up < vert_dist_down] = 3
        # Assumes no coin spawns under agent
        #assert torch.logical_and(horiz_dist_right == horiz_dist_left, vert_dist_down == vert_dist_up).sum() == 0

        return actions.long()

class SymmetricCoinGame:
    def __init__(self, b, inner_ep_len, gamma_inner=0.96, grid_size: int = 3, coin_game_type: str = 'mfos'):
        if coin_game_type == 'mfos':
            self.env = CoinGameGPU(max_steps=inner_ep_len - 1, batch_size=b, grid_size=grid_size)
        elif coin_game_type == 'lola':
            self.env = LOLACoinGameGPU(max_steps=inner_ep_len - 1, batch_size=b, grid_size=grid_size)
        else:
            raise NotImplementedError

        self.inner_ep_len = inner_ep_len
        self.b = b

    def reset(self):
        self.env_states = self.env.reset()[0]
        self.rewards_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
        self.rewards_outer = torch.Tensor(np.array([0.0] * self.b)).to(device)
        self.dones_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
        self.dones_outer = torch.Tensor(np.array([0.0] * self.b)).to(device)
        return self._prep_state()

    def _prep_state(self):

        rewards_inner_tiled = torch.tile(self.rewards_inner[None, None].T, [1, self.env.grid_size, self.env.grid_size])[:, None]
        rewards_outer_tiled = torch.tile(self.rewards_outer[None, None].T, [1, self.env.grid_size, self.env.grid_size])[:, None]
        dones_inner_tiled = torch.tile(self.dones_inner[None, None].T, [1, self.env.grid_size, self.env.grid_size])[:, None]
        env_states_outer = torch.stack([self.env_states[:, 1], self.env_states[:, 0], self.env_states[:, 3], self.env_states[:, 2]], dim=1)
        return [
            torch.cat([self.env_states, rewards_inner_tiled, rewards_outer_tiled, dones_inner_tiled], axis=1),
            torch.cat([env_states_outer, rewards_outer_tiled, rewards_inner_tiled, dones_inner_tiled], axis=1),
        ]

    def step(self, actions):
        if torch.any(self.dones_inner):
            info = None
            self.env_states = self.env.reset()[0]
            self.rewards_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
            self.rewards_outer = torch.Tensor(np.array([0.0] * self.b)).to(device)
            self.dones_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
        else:
            self.env_states, rewards, self.dones_inner, info = self.env.step(actions)
            self.env_states = self.env_states[0]
            self.rewards_inner, self.rewards_outer = rewards
        return self._prep_state(), [self.rewards_inner, self.rewards_outer], info


class CoinGamePPO:
    def __init__(self, b, inner_ep_len, gamma_inner=0.96, first=False, grid_size=3, coin_game_type: str = 'mfos'):
        if coin_game_type == 'mfos':
            self.env = CoinGameGPU(max_steps=inner_ep_len - 1, batch_size=b, grid_size=grid_size)
        elif coin_game_type == 'lola':
            self.env = LOLACoinGameGPU(max_steps=inner_ep_len - 1, batch_size=b, grid_size=grid_size)
        else:
            raise NotImplementedError

        self.inner_ep_len = inner_ep_len
        self.b = b
        self.first = first

    def reset(self):
        """
        Hyperparams for inner PPO
        """
        input_shape = [4, self.env.grid_size, self.env.grid_size]
        action_dim = 4
        n_latent_var = 8
        lr = 0.005

        betas = (0.9, 0.999)
        gamma = 0.96  # discount factor
        tau = 0.3  # GAE
        K_epochs = 80  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO

        self.inner_agent = PPO(input_shape, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, tau)  # MAYBE DONT LET INNER AGENT BE RNN
        self.inner_memory = Memory()

        self.env_states = self.env.reset()
        self.rewards_inner = torch.zeros(self.b).to(device)
        self.rewards_outer = torch.zeros(self.b).to(device)
        self.dones_inner = torch.zeros(self.b).to(device)
        self.dones_outer = torch.zeros(self.b).to(device)
        self.t = 0

        return self._prep_state()

    def _prep_state(self):
        rewards_inner_tiled = torch.tile(self.rewards_inner[None, None].T, [1, self.env.grid_size, self.env.grid_size])[:, None]
        rewards_outer_tiled = torch.tile(self.rewards_outer[None, None].T, [1, self.env.grid_size, self.env.grid_size])[:, None]
        dones_inner_tiled = torch.tile(self.dones_inner[None, None].T, [1, self.env.grid_size, self.env.grid_size])[:, None]

        return torch.cat([self.env_states[0], rewards_inner_tiled, rewards_outer_tiled, dones_inner_tiled], axis=1)

    def step(self, actions):
        self.t += 1
        if torch.any(self.dones_inner):
            info = None
            assert self.t % self.inner_ep_len == 0
            self.env_states = self.env.reset()
            self.rewards_inner = torch.zeros(self.b).to(device)
            self.rewards_outer = torch.zeros(self.b).to(device)
            self.dones_inner = torch.zeros(self.b).to(device)
            self.dones_outer = torch.zeros(self.b).to(device)
            self.inner_agent.update(self.inner_memory)
            self.inner_memory.clear_memory()
        else:
            with torch.no_grad():
                self.inner_actions = self.inner_agent.policy_old.act(self.env_states[1], self.inner_memory)

                if self.first:
                    self.env_states, rewards, self.dones_inner, info = self.env.step([actions, self.inner_actions])
                    self.rewards_outer, self.rewards_inner = rewards
                else:
                    self.env_states, rewards, self.dones_inner, info = self.env.step([self.inner_actions, actions])
                    self.rewards_inner, self.rewards_outer = rewards

                self.inner_memory.rewards.append(self.rewards_inner)

        return self._prep_state(), self.rewards_outer, self.dones_outer, self.rewards_inner, info
