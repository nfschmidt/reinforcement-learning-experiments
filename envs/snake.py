import random
import numpy as np
from collections import deque
import gym
import matplotlib.pyplot as plt

class Snake(gym.Env):
    RIGHT = 0
    FORWARD = 1
    LEFT = 2

    _DIR_UP = 0
    _DIR_LEFT = 1
    _DIR_DOWN = 2
    _DIR_RIGHT = 3

    def __init__(self,
                 step_reward=0,
                 target_reward=100,
                 collision_reward=-100,
                 side=20,
                 max_steps_without_gain=None):

        super().__init__()

        # map shape
        self._side = side

        # rewards
        self._step_reward = step_reward
        self._target_reward = target_reward
        self._collision_reward = collision_reward

        # steps
        if max_steps_without_gain is not None:
            self._max_steps_without_gain = max_steps_without_gain
        else:
            self._max_steps_without_gain = self._side*3

        # action space
        self.action_space = gym.spaces.Discrete(3)

        # observation space
        self.observation_space = \
                gym.spaces.Box(low=-self._side,
                               high=self._side,
                               shape=(5,),
                               dtype=np.int8)

        self.reset()

    def render(self, mode='human', close=False):
        grid = np.zeros((self._side, self._side), dtype=np.uint8)

        # add snake
        grid[self._head[0]][self._head[1]] = 200
        for s in self._body:
            grid[s[0]][s[1]] = 128

        # add target
        grid[self._target[0]][self._target[1]] = 255

        state = self._state() 

        plt.imshow(grid, cmap='gray')
        plt.show()
        print(f'steps_without_gain: {self._steps_without_gain}')
        print(f'direction: {self._direction}')
        print(f'target: {self._target}')
        print(f'target_distance: {state[:2]}')
        print(f'obstacle_distance: {state[2:]}')

    def step(self, action):
        reward = 0

        if self._steps_without_gain >= self._max_steps_without_gain:
            self._done = True
            if not self._body:
                # punish the snake if it did not eat a target
                reward = -100

        if self._done:
            return self._state(), reward, True, None

        # this computation takes advantage of the actions and
        # directions numbering to update the direction
        self._direction = (self._direction + action - 1) % 4

        if self._direction == self.__class__._DIR_RIGHT:
            new_head = self._head[0], self._head[1]+1
        elif self._direction == self.__class__._DIR_LEFT:
            new_head = self._head[0], self._head[1]-1
        elif self._direction == self.__class__._DIR_UP:
            new_head = self._head[0]-1, self._head[1]
        elif self._direction == self.__class__._DIR_DOWN:
            new_head = self._head[0]+1, self._head[1]

        # If the head collides, end the episode
        if self._collision(new_head):
            self._done = True
            return self._state(), self._collision_reward, True, None

        old_head = self._head
        self._head = new_head

        if self._body:
            self._body.appendleft(old_head)

        if self._head == self._target:
            self._steps_without_gain = 0

            if not self._body:
                self._body.append(old_head)
            self._new_target()
            reward = self._target_reward
        else:
            self._steps_without_gain += 1

            if self._body:
                self._body.pop()
            reward = self._step_reward

        return self._state(), reward, False, None

    def _new_target(self):
        new_target = None
        while not new_target or new_target == self._head or new_target in self._body:
            new_target = tuple(random.randint(0, self._side-1) for _ in range(2))

        self._target = new_target

    def _collision(self, new_head):
        return (new_head[0] < 0 or
                new_head[0] >= self._side or
                new_head[1] < 0 or
                new_head[1] >= self._side or
                new_head in self._body or
                new_head == self._head)

    def _state(self):
        left_distance = self._head[1]
        for d in range(1, self._head[1]+1):
            if (self._head[0], self._head[1] - d) in self._body:
                left_distance = d-1
                break

        right_distance = self._side - self._head[1] - 1
        for d in range(1, self._side - self._head[1]):
            if (self._head[0], self._head[1] + d) in self._body:
                right_distance = d-1
                break

        up_distance = self._head[0]
        for d in range(1, self._head[0]+1):
            if (self._head[0] - d, self._head[1]) in self._body:
                up_distance = d-1
                break

        down_distance = self._side - self._head[0] - 1
        for d in range(1, self._side - self._head[0]):
            if (self._head[0] + d, self._head[1]) in self._body:
                down_distance = d-1
                break

        if self._direction == self.__class__._DIR_UP:
            forward_distance = up_distance
            left_distance = left_distance
            right_distance = right_distance
        elif self._direction == self.__class__._DIR_LEFT:
            forward_distance = left_distance
            left_distance = down_distance
            right_distance = up_distance
        elif self._direction == self.__class__._DIR_DOWN:
            forward_distance = down_distance
            left_distance = right_distance
            right_distance = left_distance
        elif self._direction == self.__class__._DIR_RIGHT:
            forward_distance = right_distance
            left_distance = up_distance
            right_distance = down_distance

        target_distance = \
                self._target[0] - self._head[0], self._target[1] - self._head[1]

        return (target_distance[0], target_distance[1],
                left_distance, forward_distance, right_distance)

    def reset(self):
        self._head = tuple(random.randint(0, self._side-1) for _ in range(2))
        self._body = deque()
        self._direction = random.randint(0, 3)

        # set target position
        self._new_target()

        # steps
        self._steps_without_gain = 0

        # done
        self._done = False
