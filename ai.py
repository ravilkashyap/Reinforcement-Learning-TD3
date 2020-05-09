# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from collections import namedtuple

img_size = 30

Action = namedtuple("action", ["rotation", "velocity"])

# Helper function to add state space
class StateParam:
    def __init__(self):
        self.image = []
        self.distance = []
        self.pos_orientation = []
        self.neg_orientation = []

    def add(self, state):
        self.image.append(state.image)
        self.pos_orientation.append(state.pos_orientation)
        self.neg_orientation.append(state.neg_orientation)
        self.distance.append(state.distance)

    def convert_to_tensor(self):

        self.image = torch.from_numpy(
            np.array(self.image).reshape(len(self.image), 3, img_size, img_size)
        ).float()
        self.pos_orientation = np.array(self.pos_orientation).reshape(
            len(self.pos_orientation), 1
        )
        self.neg_orientation = np.array(self.neg_orientation).reshape(
            len(self.neg_orientation), 1
        )
        self.distance = np.array(self.distance).reshape(len(self.distance), 1)

        return (
            self.image,
            Variable(
                torch.from_numpy(
                    np.concatenate(
                        [self.pos_orientation, self.neg_orientation, self.distance], 1
                    )
                )
            ).float(),
        )

# Helper function to add action space
class ActionParam:
    def __init__(self):
        self.rotation = []
        self.velocity = []

    def add(self, action):
        self.rotation.append(action.rotation)
        self.velocity.append(action.velocity)

    def add_tensor(self, values):
        values = values.detach().numpy()
        for i in values:
            acn = Action(i[0], i[1])
            self.add(acn)

    def convert_to_value(self):
        # self.rotation = np.array(self.rotation).reshape(len(self.rotation))
        # self.velocity = np.array(self.velocity).reshape(len(self.velocity))
        return np.array([self.rotation[0], self.velocity[0]])

    def convert_to_values(self):
        self.rotation = np.array(self.rotation).reshape(len(self.rotation), 1)
        self.velocity = np.array(self.velocity).reshape(len(self.velocity), 1)
        return np.array(np.concatenate([self.rotation, self.velocity], 1))

    def convert_to_tensor(self):
        self.rotation = np.array(self.rotation).reshape(len(self.rotation), 1)
        self.velocity = np.array(self.velocity).reshape(len(self.velocity), 1)
        return Variable(
            torch.from_numpy(np.concatenate([self.rotation, self.velocity], 1))
        ).float()

# Replay buffer to store info of each timestep
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
            StateParam(),
            StateParam(),
            ActionParam(),
            [],
            [],
        )
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.add(state)
            batch_next_states.add(next_state)
            batch_actions.add(action)
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return (
            batch_states,
            batch_next_states,
            batch_actions,
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1),
        )

        # Conv class


# helper function for conv blocks 
class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), dropout=0.1, **kwargs
    ):
        super(ConvBlock, self).__init__()
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


# Creating the architecture of the Actor model

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = torch.FloatTensor(max_action)
        self.conv1 = ConvBlock(in_channels=3, out_channels=10)
        self.conv2 = ConvBlock(in_channels=10, out_channels=20)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = ConvBlock(in_channels=20, out_channels=30)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(10 + state_dim, 20)
        # self.linear1 = nn.Linear(3, 20)
        self.batchnorm2 = nn.BatchNorm1d(20)
        self.linear2 = nn.Linear(20, action_dim)

    def forward(self, image, state_param):

        # print(image.shape)
        x = self.conv1(image)
        x = self.conv2(x)
        # x = self.maxpool1(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.batchnorm1(x)
        x = self.gap(x)
        # print(x.shape)
        x = x.reshape(image.shape[0], 10)
        # print(x.shape)
        # print(or1.shape)
        # print(or2.shape)

        # x = torch.cat([x, or1, or2], 1)
        x = torch.cat([x, state_param], 1)
        # print(x.shape)
        # exit()
        x = self.linear1(x)
        # print(x.shape)
        x = self.batchnorm2(x)
        x = self.linear2(x)
        # print(x.shape)
        # print(self.max_action)
        # print()
        # exit()
        x[:, 0] = self.max_action[0] * torch.tanh(x[:, 0])
        x[:, 1] = self.max_action[1] * torch.sigmoid(x[:, 1])
        # print(x.shape)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.action_dim = action_dim

        # critic 1
        self.conv1_a = ConvBlock(in_channels=3, out_channels=10)
        self.conv2_a = ConvBlock(in_channels=10, out_channels=20)
        self.maxpool1_a = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_a = ConvBlock(in_channels=20, out_channels=30)
        self.maxpool2_a = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_a = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1))
        self.batchnorm1_a = nn.BatchNorm2d(10)
        self.gap_a = nn.AdaptiveAvgPool2d(1)
        self.linear1_a = nn.Linear(10 + action_dim + state_dim, 20)
        # self.linear1_a = nn.Linear(5, 20)
        self.batchnorm2_a = nn.BatchNorm1d(20)
        self.linear2_a = nn.Linear(20, 1)

        # critic 2
        self.conv1_b = ConvBlock(in_channels=3, out_channels=10)
        self.conv2_b = ConvBlock(in_channels=10, out_channels=20)
        self.maxpool1_b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_b = ConvBlock(in_channels=20, out_channels=30)
        self.maxpool2_b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_b = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1))
        self.batchnorm1_b = nn.BatchNorm2d(10)
        self.gap_b = nn.AdaptiveAvgPool2d(1)
        self.linear1_b = nn.Linear(10 + action_dim + state_dim, 20)
        # self.linear1_b = nn.Linear(5, 20)
        self.batchnorm2_b = nn.BatchNorm1d(20)
        self.linear2_b = nn.Linear(20, 1)

    def forward(self, image, state_param, action):
        # print("inside forward for critic")

        # print(image.shape)
        # print(action.shape)
        # print("==========")
        x = self.conv1_a(image)
        x = self.conv2_a(x)
        # x = self.maxpool1_a(x)
        # x = self.conv3_a(x)
        x = self.maxpool2_a(x)
        x = self.conv4_a(x)
        x = self.batchnorm1_a(x)
        # print(x.shape)
        x = self.gap_a(x)
        # print(x.shape)
        # x = x.squeeze()
        # print(x.shape)
        x = x.reshape(image.shape[0], 10)
        # print(x.shape)
        # print(x.shape)
        # print(action.shape)
        # x = x.view(-1, 1)
        # print(x.shape)
        # print(or1)
        # x = torch.cat([x, or1, or2, action], 1)
        x = torch.cat([x, state_param, action], 1)
        # print(x.shape)
        # print(x.shape)
        # exit()
        x = self.linear1_a(x)

        x = self.batchnorm2_a(x)
        # print(x.shape)
        x = self.linear2_a(x)
        # print(x.shape)
        # exit()

        y = self.conv1_b(image)
        y = self.conv2_b(y)
        # y = self.maxpool1_b(y)
        # y = self.conv3_b(y)
        y = self.maxpool2_b(y)
        y = self.conv4_b(y)
        y = self.batchnorm1_b(y)
        y = self.gap_b(y)
        # y = y.squeeze()
        y = y.reshape(image.shape[0], 10)
        # y = self.linear0_b(y)
        # y = y.view(-1, 1)
        # y = torch.cat([y, or1, or2, action], 1)
        y = torch.cat([y, state_param, action], 1)
        y = self.linear1_b(y)
        y = self.batchnorm2_b(y)
        y = self.linear2_b(y)

        return x, y

    def Q1(self, image, state_param, action):

        x = self.conv1_a(image)
        x = self.conv2_a(x)
        # x = self.maxpool1_a(x)
        # x = self.conv3_a(x)
        x = self.maxpool2_a(x)
        x = self.conv4_a(x)
        x = self.batchnorm1_a(x)
        x = self.gap_a(x)
        # x = x.squeeze()
        x = x.reshape(image.shape[0], 10)
        # x = torch.cat([x, or1, or2, action], 1)
        x = torch.cat([x, state_param, action], 1)
        x = self.linear1_a(x)
        x = self.batchnorm2_a(x)
        x = self.linear2_a(x)
        return x


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        self.actor.eval()
        state_wrapper = StateParam()
        state_wrapper.add(state)
        action = ActionParam()
        action.add_tensor(self.actor(*state_wrapper.convert_to_tensor()))
        return action.convert_to_value()

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

        for it in range(iterations):
            if it % 10 == 0:
                print(f"===  {it}  ===")

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            (
                batch_states,
                batch_next_states,
                batch_actions,
                batch_rewards,
                batch_dones,
            ) = replay_buffer.sample(batch_size)
            state = batch_states
            next_state = batch_next_states
            action = batch_actions
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(*next_state.convert_to_tensor())

            # next_action = next_action.reshape(next_action.shape[0], 1)
            # action = action.reshape(action.shape[0], 1)
            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = (
                torch.Tensor(batch_actions.convert_to_values())
                .data.normal_(0, policy_noise)
                .to(device)
            )
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = next_action + noise
            next_action = torch.max(
                torch.min(next_action, torch.Tensor(self.max_action)),
                -torch.Tensor(self.max_action),
            )
            # .clamp(
            #     -np.array(self.max_action), np.array(self.max_action)
            # )

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(
                *next_state.convert_to_tensor(), next_action
            )

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(
                *state.convert_to_tensor(), action.convert_to_tensor()
            )

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(
                    *state.convert_to_tensor(), self.actor(*state.convert_to_tensor())
                ).mean()
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
