import math
import random
import tkinter as tk
import json
import os

class Object:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

class Objective(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '*')

    def checkWin(self, ambiente) -> bool:
        up = ambiente.objects.get((self.x, self.y-1))
        down = ambiente.objects.get((self.x, self.y+1))
        left = ambiente.objects.get((self.x-1, self.y))
        right = ambiente.objects.get((self.x+1, self.y))
        if isinstance(up, Agent) or isinstance(down, Agent) or isinstance(left, Agent) or isinstance(right, Agent):
            return True
        return False

    def __str__(self):
        return f'Objective: {self.x} {self.y}'

class Obstacle(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '□')

    def __str__(self):
        return f'Obstacle: {self.x} {self.y}'

class Farol(Objective):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return f'Farol: {self.x} {self.y}'

class Accao:
    def act(self, agente, ambiente):
        pass

class Move(Accao):
    def __init__(self, direction):
        if direction != "right" and direction != "left" and direction != "up" and direction != "down":
            return

        if direction == "up":
            self.modifier = (0, -1)
        elif direction == "down":
            self.modifier = (0, 1)
        elif direction == "right":
            self.modifier = (1, 0)
        elif direction == "left":
            self.modifier = (-1, 0)
        else:
            return

    def act(self, agente, ambiente):
        current_location = (agente.x, agente.y)
        target_location = (agente.x + self.modifier[0], agente.y + self.modifier[1])

        if (target_location[0] >= ambiente.size or
                target_location[0] < 0 or
                target_location[1] >= ambiente.size or
                target_location[1] < 0):
            return False
        if ambiente.objects.get(target_location) is not None:
            return False

        ambiente.objects.pop(current_location)
        agente.x, agente.y = target_location
        ambiente.objects.update({target_location: agente})
        return True

class Sensor:
    def sense(self, agente, ambiente):
        pass

class CircularSensor(Sensor):
    def __init__(self, vision_range):
        if vision_range <= 0:
            print("Vision range must be positive")
            return
        self.vision_range = vision_range

    def sense(self, agente, ambiente):
        up_list = self.watch_direction(agente, ambiente, "up")
        down_list = self.watch_direction(agente, ambiente, "down")
        left_list = self.watch_direction(agente, ambiente, "left")
        right_list = self.watch_direction(agente, ambiente, "right")
        return {"up": up_list, "down": down_list, "left": left_list, "right": right_list}

    def watch_direction(self, agente, ambiente, direction):
        seen = []
        if direction == "up":
            modifier = (0, -1)
        elif direction == "down":
            modifier = (0, 1)
        elif direction == "right":
            modifier = (1, 0)
        elif direction == "left":
            modifier = (-1, 0)
        else:
            return None

        for i in range(1, self.vision_range + 1):
            x, y = (agente.x + (modifier[0] * i), agente.y + (modifier[1] * i))
            obj = ambiente.objects.get((x, y))
            if isinstance(obj, Obstacle):
                seen.append((obj.name, i))
                break
            if obj is not None: seen.append((obj.name, i))

        if not seen:
            return None
        return seen

class FarolSensor(Sensor):
    def __init__(self, farol: Farol):
        self.farol = farol
    def sense(self, agente, ambiente):
        return {"farol_delta": (abs(agente.x - self.farol.x), abs(agente.y - self.farol.y))}

class Agent(Object):
    def __init__(self, x: int, y: int, name: str):
        self.sensor = None
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
        self.sensor = sensor

    def comunica(self, de_agente):
        pass

    def executar(self, ambiente):
        observacao = ambiente.observacaoPara(self)
        self.observacao(observacao)
        accao = self.age()
        result = ambiente.agir(accao, self)
        recompensa = 0 # Default reward
        self.avalicaoEstadoAtual(recompensa)


class NeuralAgent(Agent):
    def __init__(self, x: int, y: int, name: str):
        super().__init__(x, y, name)

        self.path = []
        self.visit_counts = {}

        # VOLTAMOS AO BÁSICO: 11 Inputs.
        # A rede só precisa de saber onde estão as paredes e o objetivo.
        # Não a vamos confundir com contagens de passos.
        self.input_size = 11
        self.actions = ['up', 'down', 'left', 'right']

        self.weights = {
            'up': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)],
            'down': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)],
            'left': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)],
            'right': [random.uniform(-0.1, 0.1) for _ in range(self.input_size)]
        }

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.0 #todo 0 para os casos reais e 0.1 para testes

        self.last_state_inputs = None
        self.last_action = None
        self.last_action_idx = 4
        self.target_coords = None

        self.visit_counts[(x, y)] = 1

    def set_weights(self, weights):
        test_key = list(weights.keys())[0]
        if len(weights[test_key]) != self.input_size:
            return
        self.weights = weights

    def get_q_value(self, inputs, action):
        w = self.weights[action]
        limit = min(len(inputs), len(w))
        return sum(inputs[i] * w[i] for i in range(limit))

    def get_inputs(self, obs, ambiente):
        # Voltamos à versão "limpa" dos inputs
        inputs = [0.0] * self.input_size
        mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        # 1. Visão
        if obs:
            for direction, items in obs.items():
                idx = mapping.get(direction)
                if items and idx is not None:
                    obj_name, dist = items[0]
                    val = 1.0 / max(dist, 0.1)
                    if obj_name == '*':
                        inputs[idx] = val * 2.0
                    elif isinstance(obj_name, str) and ('Obstacle' in obj_name or obj_name == '□'):
                        inputs[idx] = -val

        # 2. GPS
        if self.target_coords:
            tx, ty = self.target_coords
            dx = tx - self.x
            dy = ty - self.y
            dist = math.sqrt(dx * dx + dy * dy) if (dx != 0 or dy != 0) else 1.0
            inputs[4] = dx / dist
            inputs[5] = dy / dist

        # 3. Bias e Memória
        inputs[6] = 1.0
        if self.last_action_idx is not None and self.last_action_idx < 4:
            inputs[7 + self.last_action_idx] = 1.0

        return inputs

    def is_safe_move(self, action, ambiente):
        mod = (0, 0)
        if action == "up":
            mod = (0, -1)
        elif action == "down":
            mod = (0, 1)
        elif action == "left":
            mod = (-1, 0)
        elif action == "right":
            mod = (1, 0)

        tx = self.x + mod[0]
        ty = self.y + mod[1]

        if not (0 <= tx < ambiente.size and 0 <= ty < ambiente.size): return False
        obj = ambiente.objects.get((tx, ty))
        if isinstance(obj, Obstacle): return False
        return True

    def age(self, ambiente) -> Accao:
        # A MÁGICA ACONTECE AQUI AGORA

        if self.last_state_inputs is None:
            safe_actions = [a for a in self.actions if self.is_safe_move(a, ambiente)]
            if not safe_actions: safe_actions = self.actions
            accao = random.choice(safe_actions)
            self.last_action = accao
            mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
            self.last_action_idx = mapping[accao]
            return Move(accao)

        scores = {}
        for action in self.actions:
            # 1. O que a rede acha (Baseado em paredes e objetivo)
            q_val = self.get_q_value(self.last_state_inputs, action)

            # 2. Segurança Hardcoded (Paredes)
            if not self.is_safe_move(action, ambiente):
                q_val = -float('inf')
            else:
                # 3. Penalidade de Tédio Hardcoded (O Segredo)
                # Calculamos onde essa ação vai dar
                mod = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}[action]
                target_pos = (self.x + mod[0], self.y + mod[1])

                # Vamos buscar quantas vezes já lá fomos
                visits = self.visit_counts.get(target_pos, 0)

                # SUBTRAÍMOS do score final.
                # A rede neural diz "Vai!", mas nós dizemos "Espera, já fui aí 10 vezes".
                # O valor 0.5 é ajustável. Se ele ainda andar muito em círculos, aumenta para 1.0 ou 2.0.
                q_val -= (visits * 50.5)

            scores[action] = q_val

        # Epsilon-Greedy
        if random.random() < self.epsilon:
            safe_actions = [a for a in self.actions if scores[a] > -float('inf')]
            best_action = random.choice(safe_actions) if safe_actions else random.choice(self.actions)
        else:
            best_action = max(scores, key=scores.get)

        self.last_action = best_action
        mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.last_action_idx = mapping[best_action]

        return Move(best_action)

    def learn(self, reward, new_inputs):
        if self.last_state_inputs is None or self.last_action is None: return

        current_q = self.get_q_value(self.last_state_inputs, self.last_action)
        max_next_q = -float('inf')
        for action in self.actions:
            q = self.get_q_value(new_inputs, action)
            if q > max_next_q: max_next_q = q

        target = reward + (self.discount_factor * max_next_q)
        error = target - current_q

        w = self.weights[self.last_action]
        new_w = []
        for i in range(len(w)):
            if i < len(self.last_state_inputs):
                delta = self.learning_rate * error * self.last_state_inputs[i]
                new_w.append(w[i] + delta)
            else:
                new_w.append(w[i])
        self.weights[self.last_action] = new_w

    def executar(self, ambiente):
        # 1. GPS
        found = False
        for pos, obj in ambiente.objects.items():
            if isinstance(obj, Objective):
                self.target_coords = pos
                found = True
                break
        if not found: self.target_coords = None

        # 2. Perceção
        observacao_bruta = ambiente.observacaoPara(self)
        self.observacao(observacao_bruta)
        current_inputs = self.get_inputs(observacao_bruta, ambiente)
        self.last_state_inputs = current_inputs

        # 3. Decisão (AGORA INCLUI A PENALIDADE DE TÉDIO INTERNAMENTE)
        accao = self.age(ambiente)

        # 4. Cálculo da Recompensa
        target_x = self.x + accao.modifier[0]
        target_y = self.y + accao.modifier[1]

        valid_move = True
        if not (0 <= target_x < ambiente.size and 0 <= target_y < ambiente.size):
            reward = -20.0
            valid_move = False
        else:
            obj = ambiente.objects.get((target_x, target_y))
            if isinstance(obj, Objective):
                reward = 50.0
            elif isinstance(obj, Obstacle):
                reward = -20.0
                valid_move = False
            else:
                # Recompensa base pequena para incentivar movimento
                reward = -0.5
                # Nota: Já não precisamos de punir TANTO aqui,
                # porque a função age() já evita a repetição.
                # Mas uma punição leve ajuda a manter o caminho curto.
                visits = self.visit_counts.get((target_x, target_y), 0)
                if visits > 0:
                    reward -= 1.0

                    # 5. Executar movimento
        if ambiente.agir(accao, self):
            self.path.append((self.x, self.y))
            count = self.visit_counts.get((self.x, self.y), 0)
            self.visit_counts[(self.x, self.y)] = count + 1

        # 6. Aprender
        new_obs = ambiente.observacaoPara(self)
        new_inputs = self.get_inputs(new_obs, ambiente)
        self.learn(reward, new_inputs)

class Ambiente:
    def __init__(self, size: int):
        self.objects = {}
        self.size = size

    def add_object(self, obj: Object):
        target_pos = (obj.x, obj.y)
        if self.objects.get(target_pos) is None:
            self.objects[target_pos] = obj

    def observacaoPara(self, agente: Agent) -> dict | None:
        if isinstance(agente, Agent):
            if agente.sensor is not None:
                return agente.sensor.sense(agente, self)
        return None

    def atualizacao(self):
        pass

    def agir(self, accao: Accao, agente: Agent):
        if isinstance(accao, Accao) and isinstance(agente, Agent):
            return accao.act(agente, self)
        return False

    # Não se usa toList() porque o GUI tem acesso direto ao mapa
    def toList(self):
        matrix = [["." for y in range(self.size)] for x in range(self.size)]
        for obj in self.objects.values():
            if matrix[obj.y][obj.x] == ".":
                matrix[obj.y][obj.x] = obj.name
        return matrix

    def print(self):
        for row in self.toList():
            print(' '.join(row))

class Simulador:
    def __init__(self, listaAgente: list, ambiente: Ambiente):
        self.listaAgentes = listaAgente
        self.ambiente = ambiente
        for agent in self.listaAgentes:
            self.ambiente.add_object(agent)

    def listarAgentes(self):
        for agent in self.listaAgentes:
            print(agent)

    def executa(self):
        objectives = []
        for obj in self.ambiente.objects.values():
            if isinstance(obj, Objective):
                objectives.append(obj)
        self.ambiente.atualizacao()
        for agent in self.listaAgentes:
            agent.executar(self.ambiente)
        
        win = False
        for obj in objectives:
            if obj.checkWin(self.ambiente):
                win = True
        return win

class SimulationGUI:
    def __init__(self, master, simulador):
        self.master = master
        self.simulador = simulador
        self.ambiente = simulador.ambiente
        self.running = False

        master.title("Simulação Multi-Agente")
        master.configure(bg='#f0f0f0')

        # Layout Principa
        self.main_frame = tk.Frame(master, padx=15, pady=15, bg='#f0f0f0')
        self.main_frame.pack(fill="both", expand=True)

        # Frame do Mapa (Esquerda)
        self.map_frame = tk.Frame(self.main_frame, padx=10, pady=10, bg='#ffffff', relief=tk.RAISED, borderwidth=2)
        self.map_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Frame de Informação (Direita)
        self.info_frame = tk.Frame(self.main_frame, padx=10, pady=10, bg='#e0e0e0', relief=tk.RIDGE, borderwidth=1)
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Grelha do Mapa
        self.cells = []
        for r in range(self.ambiente.size):
            row_cells = []
            for c in range(self.ambiente.size):
                cell = tk.Label(self.map_frame, text=".", width=4, height=2,
                                borderwidth=1, relief="ridge", font=('Arial', 10, 'bold'),
                                fg='#cccccc', bg='#ffffff')
                cell.grid(row=r, column=c, padx=1, pady=1)
                row_cells.append(cell)
            self.cells.append(row_cells)

        # Botões de Controlo
        self.control_frame = tk.Frame(self.info_frame, bg='#e0e0e0')
        self.control_frame.pack(pady=10, fill='x')

        self.start_button = tk.Button(self.control_frame, text="▶ Iniciar", command=self.start_simulation,
                                      bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'), relief=tk.RAISED)
        self.start_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.stop_button = tk.Button(self.control_frame, text="■ Parar", command=self.stop_simulation,
                                     state=tk.DISABLED, bg='#F44336', fg='white', font=('Arial', 10, 'bold'),
                                     relief=tk.RAISED)
        self.stop_button.pack(side=tk.LEFT, expand=True, padx=5)

        # Caixa de Texto para Visão
        tk.Label(self.info_frame, text="👁 Perceção (Sensores)",
                 font=('Arial', 12, 'bold'), bg='#e0e0e0', fg='#333333').pack(pady=(15, 5))

        self.vision_text = tk.Text(self.info_frame, height=20, width=35, state=tk.DISABLED,
                                   font=('Consolas', 9), bg='#ffffff', fg='#000000', relief=tk.FLAT)
        self.vision_text.pack(pady=5, fill='both', expand=True)

        self.update_gui()

    def update_gui(self):
        for r in range(self.ambiente.size):
            for c in range(self.ambiente.size):
                obj = self.ambiente.objects.get((c, r))  # Nota: get((x, y))

                text = "."
                bg_color = '#ffffff'
                fg_color = '#eeeeee'

                if obj:
                    text = obj.name
                    if isinstance(obj, Agent):
                        bg_color = '#4DD0E1'  # Azul
                        fg_color = 'black'
                    elif isinstance(obj, Obstacle):
                        bg_color = '#757575'  # Cinzento
                        fg_color = 'white'
                    elif isinstance(obj, Objective):
                        bg_color = '#8BC34A'  # Verde
                        fg_color = 'black'

                self.cells[r][c].config(text=text, bg=bg_color, fg=fg_color,
                                        relief='raised' if obj else 'ridge')

        # 2. Atualizar Texto de Perceção
        self.vision_text.config(state=tk.NORMAL)
        self.vision_text.delete(1.0, tk.END)

        for agent in self.simulador.listaAgentes:
            self.vision_text.insert(tk.END, f"Agente {agent.name}:\n", 'header')

            obs = self.ambiente.observacaoPara(agent)
            if obs:
                for direction, items in obs.items():
                    self.vision_text.insert(tk.END, f"  {direction}: ")
                    if items:
                        # items is list of (name, dist)
                        display_items = [f"{name}({dist})" for name, dist in items]
                        self.vision_text.insert(tk.END, f"{', '.join(display_items)}\n")
                    else:
                        self.vision_text.insert(tk.END, "-\n", 'dim')
            else:
                self.vision_text.insert(tk.END, "  (Sem dados)\n", 'dim')
            self.vision_text.insert(tk.END, "\n")

        self.vision_text.tag_config('header', foreground='blue', font=('Arial', 10, 'bold'))
        self.vision_text.tag_config('dim', foreground='gray')
        self.vision_text.config(state=tk.DISABLED)

    def run_step(self):
        if self.running:
            win = self.simulador.executa()
            self.update_gui()
            if win:
                print("Simulation Won!")
                self.stop_simulation()
            else:
                self.master.after(100, self.run_step)

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.run_step()

    def stop_simulation(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    la = []

    ENV_SCENARIOS = [
        # Scenario 0: Original
        {
            "obstacles": [
                (2, 2), (3, 2), (4, 2),
                (4, 4), (4, 5),
                (6, 2), (7, 2), (8, 2),
                (6, 4), (6, 5),
                (7, 8), (6, 8), (5, 8), (4, 8), (3, 8),
                (8, 7), (2, 7), (9,7)
            ],
            "objective": (8, 8),
            "start_pos": (0, 0)
        },
        {
            "obstacles": [
                (2, 2), (3, 2), (4, 2),
                (4, 4), (4, 5),
                (6, 2), (7, 2), (8, 2),
                (6, 4), (6, 5),
                (7, 8), (6, 8), (5, 8), (4, 8), (3, 8),
                (8, 7), (2, 7), (7,9)
            ],
            "objective": (8, 8),
            "start_pos": (0, 0)
        },
        {
            "obstacles": [
                (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
                (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9),
                (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8),

                (8, 4), (8, 5), (8, 6),

                (2, 2), (3, 2), (4, 2),
                (7, 2), (8, 2),

                (3, 3),

                (6, 1), (6, 2), (6, 3), (6, 4),

                (5, 7), (6, 7), (7, 7)
            ],
            "objective": (8, 7),
            "start_pos": (1, 2)
        }
    ]

    def create_environment(scenario):
        amb = Ambiente(10)
        for obs_pos in scenario["obstacles"]:
            amb.add_object(Obstacle(obs_pos[0], obs_pos[1]))

        obj_pos = scenario["objective"]
        amb.add_object(Objective(obj_pos[0], obj_pos[1]))

        start_x, start_y = scenario["start_pos"]
        agent = NeuralAgent(start_x, start_y, "N")
        agent.instala(CircularSensor(3))
        la.append(agent)
        return amb

    amb = create_environment(ENV_SCENARIOS[2])



    simulador = Simulador(la, amb)

    root = tk.Tk()
    
    # Try to load best weights if available
    if os.path.exists("best_weights.json"):
        try:
            with open("best_weights.json", "r") as f:
                weights = json.load(f)
            for a in la:
                a.set_weights(weights)
            print("Loaded best weights!")
        except Exception as e:
            print(f"Could not load weights: {e}")

    gui = SimulationGUI(root, simulador)
    root.mainloop()