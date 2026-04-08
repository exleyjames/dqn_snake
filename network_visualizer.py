import pygame
import numpy as np

class Network_Visualizer:
    def __init__(self, screen, x_offset, model):
        self.screen = screen
        self.x_offset = x_offset
        self.model = model

    def draw_layer(self, activations, layer_index, y_spacing=20):
        activations = np.array(activations).flatten()

        max_neurons = min(len(activations), 20)
        activations = activations[:max_neurons]

        layer_height = max_neurons * 25
        screen_height = self.screen.get_height()

        start_y = (screen_height - layer_height) // 2

        x = self.x_offset + layer_index * 150

        positions = []

        for i in range(max_neurons):
            y = start_y + i * 25
            positions.append((x, y))

        return positions

    def draw(self, activations):

        layer_positions = []
        processed_activations = []

        # STEP 1: compute positions only
        for i, layer in enumerate(activations):
            layer = np.array(layer).flatten()
            layer = layer[:20]

            processed_activations.append(layer)
            positions = self.draw_layer(layer, i)
            layer_positions.append(positions)

        # STEP 2: draw connections FIRST
        for layer_index in range(len(layer_positions) - 1):

            weights = self.model.layers[layer_index + 1].get_weights()
            if not weights:
                continue

            weight_matrix = weights[0]

            prev_positions = layer_positions[layer_index]
            next_positions = layer_positions[layer_index + 1]
            prev_activations = processed_activations[layer_index]

            max_in = min(len(prev_positions), weight_matrix.shape[0])
            max_out = min(len(next_positions), weight_matrix.shape[1])

            for i in range(max_in):
                for j in range(max_out):

                    weight = weight_matrix[i][j]
                    activation_strength = abs(prev_activations[i])

                    scaled = min(1.0, abs(weight) * activation_strength * 5)
                    intensity = int(255 * scaled)

                    if weight > 0:
                        color = (0, intensity, 0)
                    else:
                        color = (intensity, 0, 0)

                    thickness = max(1, int(3 * scaled))

                    pygame.draw.line(
                        self.screen,
                        color,
                        prev_positions[i],
                        next_positions[j],
                        thickness
                    )

        # STEP 3: draw neurons LAST (so they sit on top)
        for layer_index, layer in enumerate(processed_activations):

            for i, value in enumerate(layer):
                x, y = layer_positions[layer_index][i]

                brightness = int(255 * (1 / (1 + np.exp(-value))))
                color = (brightness, brightness, brightness)

                pygame.draw.circle(self.screen, color, (x, y), 10)