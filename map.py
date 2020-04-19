# Self Driving Car

import time
from random import randint, random, uniform
import pickle
import matplotlib.pyplot as plt

# Importing the libraries
import numpy as np
import imutils
import cv2
from PIL import Image

# Importing the Kivy packages
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Ellipse, Line
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ObjectProperty, ReferenceListProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.vector import Vector
from PIL import Image as PILImage

from kivy.core.window import Window

Window.size = (1429, 660)

from td3 import TD3, ReplayBuffer

import torch

action_dim = 1
max_action = 5

img_size = 100
state_size = (3, img_size, img_size)
policy = TD3(state_size, action_dim, max_action)


goal_x1 = 110
goal_y1 = 100

goal_x2 = 1140
goal_y2 = 100

goal_x = goal_x2
goal_y = goal_y2

replay_buffer = ReplayBuffer()
# Adding this line if we don't want the right click to put a red point
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1429")
Config.set("graphics", "height", "660")


reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")
counter = 0
car_img = cv2.imread("./images/car_L_resized.png")
citymap = cv2.imread("./images/citymap.png")[:, :, ::-1]
# sand = np.zeros((longueur,largeur))
img = PILImage.open("./images/mask.png").convert("L")
sand = np.asarray(img) / 255
# Initializing the map
first_update = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 0  # Random seed number
start_timesteps = 1000  # 1e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e3  # Total number of iterations/timesteps
save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = (
    0.2  # STD of Gaussian noise added to the actions for the exploration purposes
)
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()
episode_timesteps = 0
max_episode_steps = 100


# Initializing the last distance
last_distance = 0
episode_reward = 0


def init():

    global sand
    global goal_x
    global goal_y
    global first_update
    first_update = False
    global dest
    dest = "R"


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def add_padding(img, resize_shape):
    height, width, _ = img.shape
    to_height, to_width = resize_shape
    if width % 2 == 1:
        width_1, width_2 = width - 1, width

    else:
        width_1, width_2 = width, width

    if height % 2 == 1:
        height_1, height_2 = height - 1, height

    else:
        height_1, height_2 = height, height

    # print(width_1,width_2,height_1,height_2)

    return cv2.copyMakeBorder(
        img,
        (to_height - height_1) // 2,
        (to_height - height_2) // 2,
        (to_width - width_1) // 2,
        (to_width - width_2) // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def get_crop_coordinates(x, y):
    boundary_x, boundary_y = 1429, 660

    x_coordinates = (x - img_size // 2, x + img_size // 2)
    y_coordinates = (y - img_size // 2, y + img_size // 2)

    if x + img_size // 2 > boundary_x:
        x_coordinates = (boundary_x - img_size, boundary_x)

    if x - img_size // 2 < 0:
        x_coordinates = (0, img_size)

    if y + img_size // 2 > boundary_y:
        y_coordinates = (boundary_y - img_size, boundary_y)

    if y - img_size // 2 < 0:
        y_coordinates = (0, img_size)

    return x_coordinates, y_coordinates


def convert_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float).expand(1, -1, -1, -1).to(device)


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def reset_env(self):
        self.x = abs(randint(0, 1400))
        self.y = abs(randint(0, 600))

    def move(self, rotation):

        print(f"Rotation =  {rotation}")
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation


# Creating the game class


class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def get_state(self):

        x_car = int(self.car.x)
        y_car = int(self.car.y)

        rot_angle = -self.car.angle
        # orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0

        car_oriented = imutils.rotate_bound(car_img, -rot_angle)

        car_oriented_padded = add_padding(car_oriented, (img_size, img_size))

        x_cordinates, y_cordinates = get_crop_coordinates(x_car, y_car)

        mask_crop = np.uint8(
            sand[x_cordinates[0] : x_cordinates[1], y_cordinates[0] : y_cordinates[1]]
            * 255
        )

        mask_state = cv2.applyColorMap(mask_crop, cv2.COLORMAP_BONE)

        state_img = mask_state + car_oriented_padded

        # state_img = cv2.addWeighted(mask_state, 0.35, car_oriented_padded, 0.55, 0)

        # cv2.imwrite("si.jpg", state_img)
        # cv2.imwrite("aaaaa.jpg", mask_state+car_oriented_padded)

        cv2.imshow("img", rotate(state_img, 90))
        cv2.waitKey(1)
        # exit()

        state_img = state_img.reshape(3, img_size, img_size) / 255.0

        return state_img

    def calculate_reward(self, last_distance, dest):

        global goal_x, goal_y
        done = False
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        living_penalty = -1
        last_reward = 0
        wall_reward = 0
        dest_reward = 0

        # if outside road
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward += (
                -1
            )  # print(1, goal_x, goal_y, distance, im.read_pixel(int(self.car.x),int(self.car.y)))

        else:  # otherwise

            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))

            if distance < last_distance:
                last_reward += 0.5
            else:
                last_reward = last_reward + (-0.5)

        # wall penalty
        if (
            (self.car.x < 10)
            or (self.car.y < 10)
            or (self.car.x > 1420)
            or (self.car.y > 650)
        ):
            wall_reward = wall_reward + (-2)
            done = True

        # if close to destination
        if distance < 100:
            dest_reward = 10
            done = True
            if dest == "B":
                goal_x = goal_x2
                goal_y = goal_y2
                dest = "R"
            else:
                goal_x = goal_x1
                goal_y = goal_y1
                dest = "B"
        else:
            dest_reward += -1

        total_reward = last_reward + wall_reward + living_penalty + dest_reward
        return distance, total_reward, dest, done

    def chose_random_action(self):
        return uniform(-max_action, max_action)

    def step(self, last_distance, dest):
        return self.get_state(), self.calculate_reward(last_distance, dest)

    def update(self, dt):

        global reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global dest, episode_reward
        global counter, done, episode_timesteps
        global total_timesteps, max_action, max_episode_steps, episode_num, timesteps_since_eval, policy_freq, policy_noise, tau, discount, replay_buffer

        if first_update:
            init()

        # action=self.chose_random_action()

        # final_image=self.get_state()
        # self.car.move(action)

        # counter=counter+1
        # plt.imsave(f"./car_states/car_state{str(counter)}.png",final_image,cmap="gray")

        obs = self.get_state()
        # print(obs)
        # We start the main loop over 500,000 timesteps
        if total_timesteps < max_timesteps:

            # If the episode is done
            if done:

                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print(
                        "Total Timesteps: {} Episode Num: {} Reward: {}".format(
                            total_timesteps, episode_num, episode_reward
                        )
                    )
                    # policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                    policy.train(
                        replay_buffer,
                        min(episode_timesteps, 1),
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                    )

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    print("Saving the model")
                    timesteps_since_eval %= eval_freq
                    # evaluations.append(evaluate_policy(policy))
                    file_name = "car_t3d" + str(total_timesteps)
                    policy.save(file_name, directory="./pytorch_models")
                    # np.save("./results/%s" % (file_name), evaluations)

                # When the training step is done, we reset the state of the environment
                _ = self.car.reset_env()
                obs = self.get_state()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
            # action = env.action_space.sample()
            # action=action2rotation[randint(0,2)]
            action = self.chose_random_action()

        else:  # After 10000 timesteps, we switch to the model
            # action = policy.select_action(np.array(obs))
            action = policy.select_action(convert_to_tensor(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=1)).clip(-5, 5)
                action = float(action[0])

        self.car.move(action)

        # The agent performs the action in the environment, then reaches the next state and receives the reward
        # new_obs, reward, done, _ = env.step(action)

        new_obs, (distance, reward, dest, done) = self.step(last_distance, dest)

        last_distance = distance
        print(
            f"iterations,distance,last_reward,dest{total_timesteps,distance,reward,dest}"
        )

        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
        # if total_timesteps%1000==0:
        #     done=True
        # We increase the total reward
        episode_reward += reward
        # if episode_reward < -10000:
        #     done = True

        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        # if len(replay_buffer.storage)==1000:
        #     with open("replay_buffer.pkl","wb") as f:
        #         print("Saving===n\n\n\=========")
        #         pickle.dump(replay_buffer.storage,f)

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


# Adding the API Buttons (clear, save and load)


class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        savebtn = Button(text="save")
        savebtn.bind(on_release=self.save)
        parent.add_widget(savebtn)
        return parent

    def save(self, obj):
        print("saving brain...")
        file_name = "car_t3d_" + str(total_timesteps)
        policy.save(file_name, directory="./pytorch_models")

    def load(self, obj):
        print("loading last saved brain...")


# Running the whole thing

if __name__ == "__main__":

    CarApp().run()
