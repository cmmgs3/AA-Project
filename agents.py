import math
import random
from helpers import Accao, Move, Sensor
from world import Object

class Agent(Object):
    def __init__(self, x: int, y: int, name: str):
        self.sensor = []
        self.ultima_observacao = None
        super().__init__(x, y, name)

    def observacao(self, obs: dict):
        self.ultima_observacao = obs

    def printObservacao(self):
        if self.ultima_observacao is not None:
            print(self.name + " observacao")
            for key, value in self.ultima_observacao.items():
                print(key + ": " + str(value))

    def age(self) -> Accao:
        directions = ["up", "down", "left", "right"]
        return Move(random.choice(directions))

    def avalicaoEstadoAtual(self, recompensa: int):
        pass

    def instala(self, sensor: Sensor):
        self.sensor.append(sensor)

    def comunica(self, de_agente):
        pass

    def executar(self, ambiente):
        observacao = ambiente.observacaoPara(self)
        self.observacao(observacao)
        accao = self.age()
        result = ambiente.agir(accao, self)
        recompensa = 0  # Default reward
        self.avalicaoEstadoAtual(recompensa)


class NeuralAgent(Agent):
    def __init__(self, x: int, y: int, name: str):
        super().__init__(x, y, name)

        self.path = []
        self.visit_counts = {}

        # OPTIMIZED INPUTS
        # 0-3: Vision (Up, Down, Left, Right)
        # 4: bias (constant 1.0)
        self.input_size = 5
        self.actions = ['up', 'down', 'left', 'right']

        # Weights initialized for 4 inputs
        self.weights = {
            'up': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)],
            'down': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)],
            'left': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)],
            'right': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)]
        }

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.0

        self.last_state_inputs = None
        self.last_direction = None

        self.farol_coord = None
        self.visit_counts[(x, y)] = 1

        self.recent_positions = []
        self.max_memory = 10

        # Track last-seen objective (direction, distance) from sensors
        # This is local memory derived only from `ultima_observacao` (no global knowledge)
        self.last_seen_goal = None  # (direction_str, distance_int) or None

    def set_weights(self, weights):
        # Validation to ensure shape matches expected network
        if not isinstance(weights, dict) or len(weights) == 0:
            print("Invalid weights format (expected non-empty dict)")
            return

        # Ensure all expected actions exist
        for a in self.actions:
            if a not in weights:
                print(f"Weights missing action '{a}' - aborting load")
                return

        # Ensure each weight vector has the correct input size
        for a, vec in weights.items():
            if not isinstance(vec, list) or len(vec) != self.input_size:
                print(f"Weight mismatch for action '{a}': expected length {self.input_size}, got {len(vec) if isinstance(vec, list) else 'invalid'}")
                return

        self.weights = weights

    def get_q_value(self, inputs, action):
        w = self.weights[action]
        return sum(inputs[i] * w[i] for i in range(min(len(inputs), len(w))))

    def get_inputs(self, obs):
        # Initialize 4 vision inputs + bias (last index)
        inputs = [0.0] * (self.input_size - 1)
        mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        # We ONLY care about Vision now.
        # The Heuristic handles direction, the Eyes handle obstacles.
        if obs:
            for direction, items in obs.items():
                idx = mapping.get(direction)
                if items and idx is not None:
                    # items is expected to be a list of (name, dist)
                    try:
                        obj_name, dist = items[0]
                    except Exception:
                        # If the sensor returned an unexpected format, skip
                        continue
                    # We only really care about obstacles now,
                    # but seeing the goal (*) is still a nice bonus signal.
                    val = 1.0 / max(dist, 0.1)
                    if obj_name == '*':
                        inputs[idx] = val * 2.0
                    elif obj_name == '□':
                        inputs[idx] = -val
        # append bias feature so weights can change even when no objects are seen
        inputs.append(1.0)
        return inputs

    # Note: accept ambiente optionally so the agent can penalize invalid moves
    def age(self, ambiente=None) -> Accao:
        if self.last_state_inputs is None:
            direction = random.choice(self.actions)
            self.last_direction = direction
            return Move(direction)

        scores = {}

        for action in self.actions:
            # 1. Ask the Tiny Brain (Vision Only)
            # "Is there a wall?" -> Brain returns negative value
            q_val = self.get_q_value(self.last_state_inputs, action)

            mod = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}[action]
            target_pos = (self.x + mod[0], self.y + mod[1])

            # If we have access to the environment, mark invalid/off-grid or blocked moves very low
            if ambiente is not None:
                # Check bounds
                if (target_pos[0] < 0 or target_pos[0] >= ambiente.size or
                        target_pos[1] < 0 or target_pos[1] >= ambiente.size):
                    q_val = -float('inf')
                    scores[action] = q_val
                    continue

                # Check collision (cannot move into occupied cell)
                obj_at_target = ambiente.objects.get(target_pos)
                if obj_at_target is not None:
                    # It's occupied (obstacle, another agent, objective, etc.) -> invalid move
                    q_val = -float('inf')
                    scores[action] = q_val
                    continue

            # 2. Ask the GPS (Heuristic)
            # "Is this closer to the goal?" -> GPS adds massive points
            if self.farol_coord:
                current_dist = math.hypot(
                    self.farol_coord[0] - self.x,
                    self.farol_coord[1] - self.y
                )
                new_dist = math.hypot(
                    self.farol_coord[0] - target_pos[0],
                    self.farol_coord[1] - target_pos[1]
                )
                # We make the heuristic dominant
                dist_delta = current_dist - new_dist
                q_val += dist_delta * 5.0

            visits = self.visit_counts.get(target_pos, 0)
            # small discouragement for revisiting same cell; don't dominate learned Q
            q_val -= visits * 0.5
            if target_pos in self.recent_positions:
                q_val -= 1.0

            scores[action] = q_val

        if random.random() < self.epsilon:
            valid_actions = [a for a in self.actions if scores.get(a, -float('inf')) > -float('inf')]
            best_action = random.choice(valid_actions) if valid_actions else random.choice(self.actions)
        else:
            # pick max but ignore -inf entries (invalid moves)
            valid_scores = {a: s for a, s in scores.items() if s > -float('inf')}
            if not valid_scores:
                best_action = random.choice(self.actions)
            else:
                best_action = max(valid_scores, key=valid_scores.get)

        self.last_direction = best_action
        return Move(best_action)

    def learn(self, reward, new_inputs):
        # Ensure we have a previous state and action
        if self.last_state_inputs is None or self.last_direction is None:
            return

        if self.last_direction not in self.weights:
            return

        current_q = self.get_q_value(self.last_state_inputs, self.last_direction)
        max_next_q = max(self.get_q_value(new_inputs, a) for a in self.actions)

        target = reward + self.discount_factor * max_next_q
        error = target - current_q

        # Stabilize updates: clip error magnitude
        if error > 100.0:
            error = 100.0
        elif error < -100.0:
            error = -100.0

        for i in range(len(self.weights[self.last_direction])):
            if i < len(self.last_state_inputs):
                # weight update (gradient-like)
                self.weights[self.last_direction][i] += self.learning_rate * error * self.last_state_inputs[i]

                # small L2 regularization to avoid runaway weights
                self.weights[self.last_direction][i] *= (1.0 - 1e-4 * self.learning_rate)

                # clip weights to a reasonable range
                if self.weights[self.last_direction][i] > 5.0:
                    self.weights[self.last_direction][i] = 5.0
                elif self.weights[self.last_direction][i] < -5.0:
                    self.weights[self.last_direction][i] = -5.0

        # Move to the new state for the next step so subsequent learns use updated context
        if isinstance(new_inputs, list) and len(new_inputs) == self.input_size:
            self.last_state_inputs = new_inputs
        else:
            # If new_inputs are malformed, clear last_state to avoid incorrect learning
            self.last_state_inputs = None

    def executar(self, ambiente):
        obs = ambiente.observacaoPara(self)
        self.observacao(obs)

        if self.ultima_observacao and "farol_coord" in self.ultima_observacao:
            self.farol_coord = self.ultima_observacao["farol_coord"]
        else:
            self.farol_coord = None

        # Use only local observations for shaping rewards (no global objective coordinates)
        current_inputs = self.get_inputs(obs)
        self.last_state_inputs = current_inputs

        # record whether we see the objective before acting
        prev_seen_dist = None
        if self.ultima_observacao:
            for dir_items in self.ultima_observacao.values():
                # Some sensors (FarolSensor) return non-list values; only iterate lists
                if isinstance(dir_items, list) and dir_items:
                    for entry in dir_items:
                        # entry expected (name, dist)
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            name, dist = entry[0], entry[1]
                            if name == '*':
                                prev_seen_dist = dist
                                break
                if prev_seen_dist is not None:
                    break

        # Act
        accao = self.age(ambiente)

        if ambiente.agir(accao, self):
            self.path.append((self.x, self.y))
            self.visit_counts[(self.x, self.y)] = self.visit_counts.get((self.x, self.y), 0) + 1
            self.recent_positions.append((self.x, self.y))
            if len(self.recent_positions) > self.max_memory:
                self.recent_positions.pop(0)
            reward = -0.01  # small step cost (reduced)
        else:
            reward = -0.2  # reduced penalty for invalid move

        # After acting, observe again and shape rewards only from sensors
        new_obs = ambiente.observacaoPara(self)
        new_seen_dist = None
        if new_obs:
            for dir_items in new_obs.values():
                if isinstance(dir_items, list) and dir_items:
                    for entry in dir_items:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            name, dist = entry[0], entry[1]
                            if name == '*':
                                new_seen_dist = dist
                                break
                if new_seen_dist is not None:
                    break

        # Reward for seeing the objective (closer = larger)
        if new_seen_dist is not None:
            reward += 4.0 / max(new_seen_dist, 1)  # lower scaling
            if new_seen_dist == 1:
                reward += 15.0  # lower adjacency bonus

        # Reward if our observed distance to object decreased
        if prev_seen_dist is not None and new_seen_dist is not None and new_seen_dist < prev_seen_dist:
            reward += (prev_seen_dist - new_seen_dist) * 1.5

        # update last_seen_goal from sensors only
        self.last_seen_goal = new_seen_dist

        new_inputs = self.get_inputs(new_obs)
        self.learn(reward, new_inputs)

