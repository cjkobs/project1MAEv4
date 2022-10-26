# overhead
import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
# environment parameters
FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL = 0.18  # thrust constant

# # the following parameters are not being used in the sample code
PLATFORM_WIDTH = 0.25  # landing platform width
PLATFORM_HEIGHT = 0.06  # landing platform height
ROTATION_ACCEL = 20  # rotation constant
ROCKET_DIAMETER = 0.05  # rocket width
ROCKET_HEIGHT = ROCKET_DIAMETER * 15  # rocket height
C_D = math.pi / 16. * ROCKET_HEIGHT * ROCKET_DIAMETER * 1.  # drag coefficient
# the drag was calculated using a formula I found online for drag that stated it
# was proportional to the square of velocity and I ran with it
# define system dynamics
# Notes:
class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()
    @staticmethod
    def forward(state, action):
        """"
        action[0] = thrust controller
        action[1] = omega controller
        state[0] = x
        state[1] = y
        state[2] = x_dot
        state[3] = y_dot
        state[4] = theta
        """
        # Apply gravity

        delta_state_gravity = t.tensor([0., -GRAVITY_ACCEL * FRAME_TIME, 0., 0., 0.])

        # Thrust with quadratic drag added on, should match up with the action vectors I think, this part
        # wasn't entirely clear but I'm trying my damnedest

        state_tensor = t.zeros((5, 2))
        state_tensor[0, 0] = -1/2 * FRAME_TIME * t.sin(state[4])
        state_tensor[1, 0] = 1/2 * FRAME_TIME * t.cos(state[4])
        state_tensor[2, 0] = -t.sin(state[4]) - C_D * state[2] ** 2
        state_tensor[3, 0] = t.cos(state[4]) - C_D * state[3] ** 2
        state_tensor[4, 1] = 1/FRAME_TIME
        delta_state = BOOST_ACCEL * FRAME_TIME * t.matmul(action, t.t(state_tensor))

        # Update velocity
        state = state + delta_state + delta_state_gravity
        # Update state, make sure the step_mat aligns with how your states are

        step_mat = t.tensor([[1., 0., FRAME_TIME, 0., 0.],
                             [0., 1., 0., FRAME_TIME, 0.],
                             [0., 0., 1., 0., 0.],
                             [0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 1.]])
        state = t.matmul(step_mat, state.t())

        return state

# a deterministic controller
# I tried messing around with it but everytime I tried to add a new thing in the network it would give
# me a matrix multiplication error
class Controller(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Sigmoid(),
            nn.Linear(dim_hidden, dim_output),
            nn.Tanh()
        )
    def forward(self, state):
        action = self.network(state)
        return action
# the simulator that rolls out x(1), x(2), ..., x(T)
class Simulation(nn.Module):
    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []
    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        state = [0., 3/2., 1/4., 1., 1/7.]  # initial batch of state
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        termination_error = 2 * state[0] ** 2 + 2 * (state[1] - PLATFORM_HEIGHT) ** 2 + state[2] ** 2 + state[3] ** 2 + 1 / 2 * state[4] ** 2
        return termination_error
        # added some weights to the positions as well as decreased the weight for angle

# set up the optimizer
class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.1)
    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return closure()
    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()
    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0]
        y = data[:, 1]
        x_dot = data[:, 2]
        y_dot = data[:, 3]
        theta = data[:, 4]
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(x, y)
        axs[0].set_xlabel('Horizontal Position')
        axs[0].set_ylabel('Vertical Position')
        axs[1].plot(y, x_dot, label="Horizontal Speed")
        axs[1].plot(y, y_dot, label="Vertical Speed")
        axs[1].legend(loc="best")
        axs[1].set_xlabel("Vertical Position")
        axs[1].set_ylabel("Velocity")
        axs[2].plot(y, theta, label='Angle')
        axs[2].legend(loc="best")
        axs[2].set_xlabel("Vertical Position")
        axs[2].set_ylabel("Angle")
        plt.show()
# Now it's time to run the code!

T = 80  # number of time steps
dim_input = 5  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(20)  # solve the optimization problem, default: 20