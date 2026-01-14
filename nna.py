import random
from helpers import Accao, Move
from agents import Agent

class NewNeuralAgent(Agent):
    """Neural agent with 3 internal (hidden) layers per action.

    Architecture: 4 inputs -> hidden1 -> hidden2 -> hidden3 -> 1 output
    There is one small network per action (so the agent still evaluates each
    action separately and returns a scalar Q-value). Training uses a simple
    TD(0)-style target and gradient descent on mean-squared error.
    """
    def __init__(self, x: int, y: int, name: str,
                 hidden_sizes=(16, 12, 8), learning_rate=0.01, discount_factor=0.95, epsilon=0.1):
        # Initialize base Agent (not NeuralAgent) to fully detach from NeuralAgent
        super().__init__(x, y, name)

        # Override NeuralAgent defaults where needed
        self.input_size = 4  # explicit: four perceptual inputs (no bias here)
        self.actions = ['up', 'down', 'left', 'right']

        # network architecture per action
        self.hidden_sizes = list(hidden_sizes)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Build one small MLP per action. Each network is a list of layers.
        # Each layer: {'w': [[..]], 'b': [..]} where w is out_dim x in_dim
        self.networks = {a: self._build_network(self.input_size, self.hidden_sizes + [1]) for a in self.actions}

        # training state
        self.last_state_inputs = None
        self.last_action = None

    def _build_network(self, in_dim, layer_sizes):
        net = []
        prev = in_dim
        for size in layer_sizes:
            # Initialize small random weights
            w = [[random.uniform(-0.1, 0.1) for _ in range(prev)] for _ in range(size)]
            b = [0.0 for _ in range(size)]
            net.append({'w': w, 'b': b})
            prev = size
        return net

    def _forward(self, net, inputs):
        """Forward pass. Returns (output_scalar, activations_list) where activations_list
        includes activations per layer (including input as first element) used for backprop."""
        acts = [inputs[:]]
        a = inputs[:]
        for i, layer in enumerate(net):
            w = layer['w']
            b = layer['b']
            out = []
            for row_idx in range(len(w)):
                s = b[row_idx]
                row = w[row_idx]
                # dot product
                for j in range(len(row)):
                    s += row[j] * a[j]
                out.append(s)
            # activation: ReLU for hidden, linear for last
            if i < len(net) - 1:
                a = [x if x > 0 else 0.0 for x in out]
            else:
                # final layer -> single scalar (no activation)
                a = out
            acts.append(a)
        # final output is scalar (last layer size == 1)
        return (acts[-1][0], acts)

    def _backprop_update(self, net, acts, target, lr):
        """Backpropagate error for a single-sample MSE loss on the scalar output.
        Updates weights in-place using simple SGD.
        acts: list of activations per layer (input, hidden1, ..., output)
        """
        # compute dL/dout where L = 0.5*(out - target)^2
        out = acts[-1][0]
        d_out = out - target  # derivative of MSE

        # propagate backwards
        d_next = [d_out]  # gradient wrt pre-activation of last layer (scalar)

        for layer_idx in range(len(net)-1, -1, -1):
            layer = net[layer_idx]
            a_prev = acts[layer_idx]  # activation of previous layer
            w = layer['w']
            b = layer['b']

            out_dim = len(w)
            in_dim = len(a_prev)

            # gradients for weights and biases
            # d_next is a list of length out_dim (gradients wrt pre-activation)
            dW = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
            dB = [0.0 for _ in range(out_dim)]

            for i in range(out_dim):
                dB[i] = d_next[i]
                for j in range(in_dim):
                    dW[i][j] = d_next[i] * a_prev[j]

            # gradient wrt inputs to this layer (for previous layer)
            d_prev = [0.0 for _ in range(in_dim)]
            for j in range(in_dim):
                s = 0.0
                for i in range(out_dim):
                    s += d_next[i] * w[i][j]
                d_prev[j] = s

            # update weights and biases (SGD)
            for i in range(out_dim):
                for j in range(in_dim):
                    layer['w'][i][j] -= lr * dW[i][j]
                layer['b'][i] -= lr * dB[i]

            # if there is a previous layer, apply activation derivative (ReLU)
            if layer_idx > 0:
                # derivative of ReLU on pre-activation: prev_activation > 0 -> 1 else 0
                pre_act = acts[layer_idx]  # this was pre-activation out in our forward implementation
                # But note: in our forward we used out then applied ReLU to get acts[layer_idx]
                # The stored acts[layer_idx] is post-activation; to obtain mask use it directly.
                mask = [1.0 if v > 0 else 0.0 for v in pre_act]
                # multiply element-wise
                d_next = [d_prev[k] * mask[k] for k in range(len(d_prev))]
            else:
                d_next = d_prev

    def predict_q(self, inputs):
        """Return a dict action -> q-value for the given inputs."""
        qs = {}
        for a, net in self.networks.items():
            q, _ = self._forward(net, inputs)
            qs[a] = q
        return qs

    def get_inputs(self, obs):
        """Aggregate the full observation lists into 4 directional signals.

        Returns a list of 4 floats in the order: [up, down, left, right].
        Each directional signal sums contributions from every object seen in that
        direction. Goals ('*') contribute a positive inverse-distance signal,
        obstacles ('□') and unknown objects contribute negative inverse-distance
        signals. Values are clipped to a reasonable range to keep training stable.
        """
        signals = [0.0, 0.0, 0.0, 0.0]
        mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        if not obs:
            return signals

        for direction, items in obs.items():
            idx = mapping.get(direction)
            if idx is None:
                continue
            if not items:
                continue
            try:
                for entry in items:
                    if not entry:
                        continue
                    # Expect entry like (name, dist)
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        name, dist = entry[0], entry[1]
                    else:
                        # Unexpected format: skip
                        continue

                    # Inverse-distance signal (avoid division by tiny numbers)
                    inv = 1.0 / max(dist, 0.1)

                    if name == '*':
                        weight = 2.0  # strong positive signal for goal
                    elif name == '□':
                        weight = -1.5  # stronger negative signal for obstacles
                    else:
                        weight = -0.8  # mild negative for other objects

                    signals[idx] += weight * inv
            except Exception:
                # Defensive: ignore malformed sensor entries
                continue

        # Clip signals to a stable range
        for i in range(len(signals)):
            if signals[i] > 5.0:
                signals[i] = 5.0
            elif signals[i] < -5.0:
                signals[i] = -5.0

        return signals

    def age(self, ambiente=None) -> Accao:
        """Choose action using epsilon-greedy over predicted Q-values."""
        if self.last_state_inputs is None:
            # random warm-start
            action = random.choice(self.actions)
            self.last_action = action
            return Move(action)

        qs = self.predict_q(self.last_state_inputs)
        # filter invalid moves if ambiente provided
        if ambiente is not None:
            for action in list(qs.keys()):
                mod = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}[action]
                target_pos = (self.x + mod[0], self.y + mod[1])
                if (target_pos[0] < 0 or target_pos[0] >= ambiente.size or
                        target_pos[1] < 0 or target_pos[1] >= ambiente.size):
                    qs[action] = -float('inf')
                elif ambiente.objects.get(target_pos) is not None:
                    qs[action] = -float('inf')

        if random.random() < self.epsilon:
            valid = [a for a, v in qs.items() if v > -float('inf')]
            action = random.choice(valid) if valid else random.choice(self.actions)
        else:
            # choose argmax
            valid = {a: v for a, v in qs.items() if v > -float('inf')}
            if not valid:
                action = random.choice(self.actions)
            else:
                action = max(valid, key=valid.get)

        self.last_action = action
        return Move(action)

    def learn(self, reward, new_inputs):
        """Perform a TD(0)-like update on the network corresponding to the last action.
        target = reward + gamma * max_a' Q(new_inputs, a')
        """
        if self.last_state_inputs is None or self.last_action is None:
            return

        # current predicted q for last (state, action)
        net = self.networks[self.last_action]
        cur_q, acts = self._forward(net, self.last_state_inputs)

        # estimate max next q
        next_qs = self.predict_q(new_inputs)
        max_next = max(next_qs.values()) if next_qs else 0.0

        target = reward + (0.0 if reward is None else self.gamma * max_next)

        # backprop and update only the network for last_action
        self._backprop_update(net, acts, target, self.lr)

        # move to new state
        self.last_state_inputs = new_inputs

    def executar(self, ambiente):
        """High-level loop compatible with other agents: sense -> choose -> act -> learn"""
        obs = ambiente.observacaoPara(self)
        self.observacao(obs)

        inputs = self.get_inputs(obs)
        # ensure inputs length matches self.input_size
        if len(inputs) > self.input_size:
            inputs = inputs[:self.input_size]
        elif len(inputs) < self.input_size:
            inputs = inputs + [0.0] * (self.input_size - len(inputs))

        # store state for learning
        if self.last_state_inputs is None:
            self.last_state_inputs = inputs[:]

        accao = self.age(ambiente)

        moved = ambiente.agir(accao, self)

        # simple reward shaping: small step cost, bonus for seeing objective
        reward = -0.01 if moved else -0.2

        # sensing after action
        new_obs = ambiente.observacaoPara(self)
        new_inputs = self.get_inputs(new_obs)
        if len(new_inputs) > self.input_size:
            new_inputs = new_inputs[:self.input_size]
        elif len(new_inputs) < self.input_size:
            new_inputs = new_inputs + [0.0] * (self.input_size - len(new_inputs))

        # bonus for seeing goal
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
        if new_seen_dist is not None:
            reward += 3.0 / max(new_seen_dist, 1)
            if new_seen_dist == 1:
                reward += 8.0

        # learn from transition
        self.learn(reward, new_inputs)