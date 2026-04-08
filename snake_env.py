import numpy as np
import random

class SnakeEnv:
    def __init__(self, size=20):
        self.score = None
        self.done = None
        self.direction = None
        self.snake = None
        self.snake_set = None
        self.food = None
        self.steps = None
        self.size = size
        self.reset()

    def reset(self):
        center = {'x': self.size//2, 'y': self.size//2}
        self.snake = [center]
        self.snake_set = {(center['x'], center['y'])}
        self.direction = 'RIGHT'
        self.spawn_food()
        self.done = False
        self.score = 0
        self.steps = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = {
                'x': random.randint(0, self.size-1),
                'y': random.randint(0, self.size-1)
            }
            if not self.is_collision(self.food):
                break

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        self.update_direction(action)
        head = self.snake[0].copy()

        old_distance = abs(head['x'] - self.food['x']) + abs(head['y'] - self.food['y'])

        self.move_head(head)

        new_distance = abs(head['x'] - self.food['x']) + abs(head['y'] - self.food['y'])

        self.steps += 1
        if self.steps > 100 * len(self.snake):
            self.done = True

        if self.is_collision(head):
            self.done = True
            return self.get_state(), -10, True

        self.snake.insert(0, head)  # move head
        self.snake_set.add((head['x'], head['y']))

        reward = -0.01

        if new_distance < old_distance:
            reward = 0.1
        else:
            reward = -0.1

        if head['x'] == self.food['x'] and head['y'] == self.food['y']:
            reward = 10
            self.score += 1
            self.spawn_food()
        else:
            tail = self.snake.pop()  # remove tail
            self.snake_set.remove((tail['x'], tail['y']))

        return self.get_state(), reward, self.done

    def update_direction(self, action):
        # 0=straight,1=left,2=right
        directions = ['UP','RIGHT','DOWN','LEFT']
        idx = directions.index(self.direction)

        if action == 1:  # left
            idx = (idx + 3) % 4
        elif action == 2:  # right
            idx = (idx + 1) % 4

        self.direction = directions[idx]

    def move_head(self, head):
        if self.direction == 'UP':
            head['y'] -= 1
        elif self.direction == 'DOWN':
            head['y'] += 1
        elif self.direction == 'LEFT':
            head['x'] -= 1
        elif self.direction == 'RIGHT':
            head['x'] += 1

    def is_collision(self, pos):
        if pos['x'] < 0 or pos['x'] >= self.size or pos['y'] < 0 or pos['y'] >= self.size:
            return True
        if (pos['x'], pos['y']) in self.snake_set:
            return True
        return False

    def get_state(self):
        head = self.snake[0]

        # normalized food direction
        food_dx = (self.food['x'] - head['x']) / self.size
        food_dy = (self.food['y'] - head['y']) / self.size

        # danger in 3 directions
        danger_straight = self.check_danger(0)
        danger_left = self.check_danger(1)
        danger_right = self.check_danger(2)

        #current direction
        dir_up = 1.0 if self.direction == 'UP' else 0.0
        dir_down = 1.0 if self.direction == 'DOWN' else 0.0
        dir_left = 1.0 if self.direction == 'LEFT' else 0.0
        dir_right = 1.0 if self.direction == 'RIGHT' else 0.0

        return np.array([
            food_dx,
            food_dy,
            danger_straight,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right
        ], dtype=np.float32)

    def check_danger(self, relative_action):
        temp_dir = self.get_relative_direction(relative_action)
        head = self.snake[0].copy()

        if temp_dir == 'UP':
            head['y'] -= 1
        elif temp_dir == 'DOWN':
            head['y'] += 1
        elif temp_dir == 'LEFT':
            head['x'] -= 1
        elif temp_dir == 'RIGHT':
            head['x'] += 1

        return 1.0 if self.is_collision(head) else 0.0

    def get_relative_direction(self, action):
        directions = ['UP','RIGHT','DOWN','LEFT']
        idx = directions.index(self.direction)

        if action == 1:  # left
            idx = (idx + 3) % 4
        elif action == 2:  # right
            idx = (idx + 1) % 4

        return directions[idx]
