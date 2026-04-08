import numpy as np
import random
from collections import deque
import tensorflow as tf
import pygame
from snake_env import SnakeEnv
from network_visualizer import Network_Visualizer

tf.config.run_functions_eagerly(False)

STATE_SIZE = 9
ACTION_SIZE = 3
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 20000
EPISODES = 500
EPSILON = 1.0
EPSILON_MIN = 0.005
EPSILON_DECAY = 0.997
TARGET_UPDATE_FREQ = 10

RENDER = True
FPS = 20
CELL_SIZE = 20

memory = deque(maxlen=MEMORY_SIZE)

def build_model():
    inputs = tf.keras.Input(shape=(STATE_SIZE,))
    x1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x2 = tf.keras.layers.Dense(64, activation='relu')(x1)
    outputs = tf.keras.layers.Dense(ACTION_SIZE, activation=None)(x2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='mse'
    )

    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())
#for visualization
activation_model = tf.keras.Model(
    inputs = model.input,
    outputs = [layer.output for layer in model.layers]
)

def choose_action(state, epsilon):
    #choose a random move sometimes
    if np.random.rand() < epsilon:
        return random.randrange(ACTION_SIZE)

    state = np.expand_dims(state, axis=0)
    q_values = model(state, training=False)
    action = np.argmax(q_values.numpy())

    return action

@tf.function
def train_step(states, q_values):
    model.train_on_batch(states, q_values)

def replay():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states = np.array([x[0] for x in batch])
    actions = np.array([x[1] for x in batch])
    rewards = np.array([x[2] for x in batch])
    next_states = np.array([x[3] for x in batch])
    dones = np.array([x[4] for x in batch])

    # Predict all at once
    q_values = model(states, training=False).numpy()
    next_q_values = target_model(next_states, training=False).numpy()

    # Compute targets vectorized
    for i in range(BATCH_SIZE):
        if dones[i]:
            q_values[i][actions[i]] = rewards[i]
        else:
            q_values[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])

    model.train_on_batch(states, q_values)


env = SnakeEnv(10)

if RENDER:
    pygame.init()
    max_neurons = 25
    neuron_spacing = 25
    network_height = max_neurons * neuron_spacing + 200
    board_height = env.size * CELL_SIZE
    height = max(board_height, network_height)
    width = env.size * CELL_SIZE + 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('DQN Snake')
    clock = pygame.time.Clock()
    visualizer = Network_Visualizer(screen, env.size * CELL_SIZE + 50, activation_model)

def draw():
    screen.fill((0, 0, 0))

    board_offset_y = (screen.get_height() - env.size * CELL_SIZE) // 2

    for segment in env.snake:
        pygame.draw.rect(
            screen,
            (0, 255, 0),
            (segment['x'] * CELL_SIZE, segment['y'] * CELL_SIZE + board_offset_y, CELL_SIZE, CELL_SIZE)
        )

    pygame.draw.rect(
        screen,
        (255, 0, 0),
        (env.food['x'] * CELL_SIZE, env.food['y'] * CELL_SIZE + board_offset_y, CELL_SIZE, CELL_SIZE)
    )

    activations = activation_model(np.expand_dims(state, axis=0))
    visualizer.draw(activations)

    pygame.display.flip()

epsilon = EPSILON

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    if episode % 100 == 0:
        RENDER = True
        #print("Average last 100:", np.mean(scores[-100:]))
    else:
        RENDER = False

    while not done:
        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        action = choose_action(state, epsilon)
        next_state, reward, done = env.step(action)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) > BATCH_SIZE and episode % 2 == 0:
            replay()

        if RENDER:
            draw()
            clock.tick(5)

    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode: {episode + 1}, Score: {env.score}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

print("Training complete")
model.save("snake_dqn_model.keras")

if RENDER:
    pygame.quit()