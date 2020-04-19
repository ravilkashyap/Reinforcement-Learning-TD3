from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1),
        )


class conv_block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), dropout=0.1, **kwargs
    ):
        super(conv_block, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                **kwargs,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.convblock(x)


class Actor(nn.Module):
    def __init__(self, image_size, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.conv1 = conv_block(in_channels=3, out_channels=12)
        self.conv2 = conv_block(in_channels=12, out_channels=24)
        # self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(in_channels=24, out_channels=32)
        # self.conv4 = conv_block(in_channels=32, out_channels=64)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reduce = nn.Conv2d(32, 10, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.last = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        # print("inside actor forward")
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # out = self.maxpool_1(out)
        # print(out.shape)
        out = self.conv3(out)
        # out = self.conv4(out)
        out = self.maxpool_2(out)
        out = self.reduce(out)
        # print(out.shape)
        out = self.gap(out)
        # print(out.shape)
        out = self.last(out)
        out = out.view(-1, 1)
        # print(out.shape)

        return self.max_action * F.tanh(out)


class Critic(nn.Module):
    def __init__(self, image_size, action_dim):
        super(Critic, self).__init__()

        self.conv1_1 = conv_block(in_channels=3, out_channels=12)
        self.conv2_1 = conv_block(in_channels=12, out_channels=24)
        self.conv3_1 = conv_block(in_channels=24, out_channels=32)
        self.maxpool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reduce_1_1 = nn.Conv2d(32, 10, 1)
        self.gap_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.last_1_1 = nn.Conv2d(10, 1, 1)
        self.last_1_2 = nn.Linear(1 + action_dim, 10)
        self.last_1_3 = nn.Linear(10, 1)

        self.conv1_2 = conv_block(in_channels=3, out_channels=12)
        self.conv2_2 = conv_block(in_channels=12, out_channels=24)
        self.conv3_2 = conv_block(in_channels=24, out_channels=32)
        self.maxpool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reduce_2_1 = nn.Conv2d(32, 10, 1)
        self.gap_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.last_2_1 = nn.Conv2d(10, 1, 1)
        self.last_2_2 = nn.Linear(1 + action_dim, 10)
        self.last_2_3 = nn.Linear(10, 1)

    def forward(self, x, u):
        # print("inside critic forward")
        # print(x.shape)
        # print(u.shape)
        x1 = self.conv1_1(x)
        # print(x1.shape)
        x1 = self.conv2_1(x1)
        # print(x1.shape)
        x1 = self.conv3_1(x1)
        x1 = self.maxpool_1_1(x1)
        x1 = self.reduce_1_1(x1)
        # print(x1.shape)
        x1 = self.gap_1(x1)
        # print(x1.shape)
        x1 = self.last_1_1(x1)
        # print(x1.shape)
        x1 = x1.view(-1, 1)
        # print(x1.shape)
        x1 = torch.cat([x1, u], 1)
        # print(x1.shape)
        x1 = self.last_1_2(x1)
        # print(x1.shape)
        x1 = self.last_1_3(x1)

        x2 = self.conv1_2(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_2_1(x2)
        x2 = self.reduce_2_1(x2)
        x2 = self.gap_2(x2)
        x2 = self.last_2_1(x2)
        x2 = x2.view(-1, 1)
        x2 = torch.cat([x2, u], 1)
        x2 = self.last_2_2(x2)
        x2 = self.last_2_3(x2)

        return x1, x2

    def Q1(self, x, u):
        x2 = self.conv1_2(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_2_1(x2)
        x2 = self.reduce_2_1(x2)
        x2 = self.gap_2(x2)
        x2 = self.last_2_1(x2)
        x2 = x2.view(-1, 1)
        x2 = torch.cat([x2, u], 1)
        x2 = self.last_2_2(x2)
        x2 = self.last_2_3(x2)
        return x2


class TD3(object):
    def __init__(self, image_size, action_dim, max_action):
        self.actor = Actor(image_size, max_action).to(device)
        self.actor_target = Actor(image_size, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(image_size, action_dim).to(device)
        self.critic_target = Critic(image_size, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action


    def select_action(self, state):
        # state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):

        for it in tqdm(range(iterations)):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            (
                batch_states,
                batch_next_states,
                batch_actions,
                batch_rewards,
                batch_dones,
            ) = replay_buffer.sample(batch_size)
            if len(batch_actions.shape) != 2:
                batch_actions = batch_actions.reshape(batch_size, 1)

            state = torch.tensor(batch_states, dtype=torch.float).to(device)
            next_state = torch.tensor(batch_next_states, dtype=torch.float).to(device)
            action = torch.tensor(batch_actions, dtype=torch.float).to(device)
            reward = torch.tensor(batch_rewards, dtype=torch.float).to(device)
            done = torch.tensor(batch_dones, dtype=torch.float).to(device)

            print(f"==============    Iteration -  {it} ===================")

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            print(f"critic loss - {critic_loss}")

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                print(f"actor loss - {actor_loss}")
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
