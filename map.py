# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint, uniform
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from collections import namedtuple
import cv2

# Importing the Dqn object from our AI in ai.py
from ai import TD3, ReplayBuffer

State = namedtuple("state", ["image", "pos_orientation", "neg_orientation", "distance"])
Action = namedtuple("action", ["rotation", "velocity"])

# Adding this line if we don't want the right click to put a red point
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1429")
Config.set("graphics", "height", "660")
Window.size = (1429, 660)


# Initializing the map
first_update = True

img_size = 200
resize_shape = (30, 30)

state_dim = 4
action_dim = 2
max_action = [5, 2]

policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

goal_x1 = 345  # 345
goal_y1 = 300

goal_x2 = 1165  # 1165
goal_y2 = 265

goal_x = goal_x2
goal_y = goal_y2

seed = 0  # Random seed number
start_timesteps = 50  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 1e5  # Total number of iterations/timesteps
save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = (
    0.2  # STD of Gaussian noise added to the actions for the exploration purposes
)
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = (
    2  # Number of iterations to wait before the policy network (Actor model) is updated
)


total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = 0
episode_timesteps = 0
max_episode_steps = 200
dest = "R"

# Images
sand = PILImage.open("./images/mask.png").convert("RGB")
orig_sand = PILImage.open("./images/MASK1.png").convert("RGB")
arrow = PILImage.open("./images/arrow.png").convert("RGBA")


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    # sand = np.zeros((longueur, largeur))
    # img = PILImage.open("./images/mask.png").convert("L")
    # sand = np.asarray(img) / 255
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.velocity = Vector(1, 0).rotate(self.angle)
        self.pos = Vector(*self.velocity) + self.pos


# Creating the game class


class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        # self.car.velocity = Vector(6, 0)

    def random_action(self):
        rotation = uniform(-max_action[0], max_action[0])
        return Action(*np.array([rotation]))

    def reset(self):
        self.car.x = abs(randint(0, 1400))
        self.car.y = abs(randint(0, 600))
        self.car.angle = abs(randint(0, 360))

    def get_image(self):
        def get_center_coord(x, y, angle):
            # Center coord to superimpose car/arrow image to get exact overlap
            # x = x - (20 * math.cos(angle) + 10 * math.sin(angle))
            # y = y - (20 * math.sin(angle) + 10 * math.cos(angle))
            x = x - 30
            y = y - 20
            return (int(np.ceil(x)), int(np.ceil(y)))

        x = self.car.x
        y = 660 - self.car.y
        # print(f"Co-ordinates - {x,y}")
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        # print(self.car.velocity)
        # exit()
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0
        rotation = self.car.angle
        # print(
        #     f"Angle and orientation - {self.car.angle, orientation, self.car.rotation}"
        # )

        mask = orig_sand.copy()
        # mask = mask.resize((1429, 660))
        pointer = arrow.copy()

        resized_arrow = pointer.resize((60, 40))
        rotated_arrow = resized_arrow.rotate(rotation)
        mask.paste(rotated_arrow, get_center_coord(x, y, rotation), rotated_arrow)
        cropped_mask = mask.crop(
            (x - img_size / 2, y - img_size / 2, x + img_size / 2, y + img_size / 2)
        )
        opencvImage = cv2.cvtColor(np.array(cropped_mask), cv2.COLOR_RGB2BGR)
        cv2.imshow("img", opencvImage)
        cv2.waitKey(1)
        # _ = cropped_mask.save("a.jpg")
        cropped_mask = cropped_mask.resize(resize_shape)
        # _ = cropped_mask.save("b.jpg")

        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

        return np.array(cropped_mask).reshape(3, resize_shape[0], resize_shape[1])

    def get_state(self):
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        distance = distance / 1429
        image = self.get_image()
        return State(*np.array([image, orientation, -orientation, distance]))

    def step(self, action):
        self.car.move(float(action.rotation))
        new_state = self.get_state()
        reward, done = self.get_reward()
        return new_state, reward, done

    def get_reward(self):
        global done, last_distance, goal_x, goal_y, dest, episode_timesteps, max_episode_steps
        living_penalty = -0.05
        sand_reward = 0
        wall_reward = 0
        goal_reward = 0
        distance_reward = 0
        angle_reward = 0
        time_reward = 0

        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

        # if sand[int(self.car.x), int(self.car.y)] > 0:
        #     self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
        #     # last_reward = -1
        # else:  # otherwise
        #     self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        #     # last_reward = -0.2

        # Reward for going closer to boundary / sand

        def is_near_boundary():
            if (
                (self.car.x == 10)
                or (self.car.y == 10)
                or (self.car.x == 1420)
                or (self.car.y == 650)
            ):
                return True
            else:
                return False

        if self.car.x < 10:
            self.car.x = 10
            wall_reward += -1
        if self.car.y < 10:
            self.car.y = 10
            wall_reward += -1
        if self.car.x > 1420:
            self.car.x = 1420
            wall_reward += -1
        if self.car.y > 650:
            self.car.y = 650
        wall_reward += -1

        # We check if the episode is done
        if episode_timesteps + 1 == max_episode_steps:
            print("Oops! Max steps reached")
            done = True
            time_reward += -100

        # if (
        #     (self.car.x < 10)
        #     or (self.car.y < 10)
        #     or (self.car.x > 1420)
        #     or (self.car.y > 650)
        # ):

        #     wall_reward += -100
        #     done = True
        #     print("Oops! Hit the boundary")

        rotate reward
        if abs(self.car.angle) > 500:
            # done = True
            angle_reward += -10 * abs(self.car.angle) / 360
        if abs(self.car.angle) > 360 * 4:
            done = True
            angle_reward += -100
            print("Oh damn Ghoomar effect!")

        # distance rewards

        # if distance < last_distance:
        #     distance_reward += 0.1
        if round(distance, 1) == round(last_distance, 1) and is_near_boundary():
            print("stuck")
            done = True
            distance_reward += -100
        # else:
        #     distance_reward += -0.2

        distance_reward = 1 - ((distance / 1429 * 2) ** 0.4)

        # Reward for reaching destination

        if distance < 50:
            distance_reward = 500
            done = True
            print("Hurray! Reached the goal")
            if dest == "B":
                goal_x = goal_x2
                goal_y = goal_y2
                dest = "R"
            else:
                goal_x = goal_x1
                goal_y = goal_y1
                dest = "B"
        last_distance = distance

        total_reward = (
            living_penalty
            + sand_reward
            + wall_reward
            + distance_reward
            + angle_reward
            + time_reward
        )
        return total_reward, done

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global total_timesteps, done, episode_num, timesteps_since_eval, episode_reward, episode_timesteps, max_episode_steps

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        state = self.get_state()

        # We start the main loop over 500,000 timesteps
        if total_timesteps < max_timesteps:

            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print(
                        "Total Timesteps: {} Episode Num: {} Episode Steps: {} Reward: {}".format(
                            total_timesteps,
                            episode_num,
                            episode_timesteps,
                            episode_reward,
                        )
                    )
                    policy.train(
                        replay_buffer,
                        100,  # episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                    )

                # When the training step is done, we reset the state of the environment
                self.reset()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = self.random_action()
                # print(f"action = {action}")
            else:  # After 10000 timesteps, we switch to the model
                action = policy.select_action(state)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action.rotation = action.rotation + np.random.normal(
                        0, expl_noise, size=action_dim
                    )
                    action.rotation = float(
                        action.rotation.clip(-max_action[0], max_action[0])
                    )

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_state, reward, done = self.step(action)

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((state, new_state, action, reward, done))

            # We increase the total reward
            episode_reward += reward

            if total_timesteps % 10 == 0 or done:
                print(
                    total_timesteps,
                    "|",
                    round(state.distance * 1429, 2),
                    ",",
                    # state[1],
                    round(state.pos_orientation, 2),
                    "|",
                    round(action.rotation, 2),
                    "|",
                    round(reward, 2),
                    "|",
                    dest,
                    round(self.car.angle, 2),
                    "|",
                    (round(self.car.x, 2), round(self.car.y, 2)),
                )

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            state = new_state
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1


# Adding the API Buttons (clear, save and load)


class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        savebtn = Button(text="save", pos=(parent.width, 0))
        loadbtn = Button(text="load")
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
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
