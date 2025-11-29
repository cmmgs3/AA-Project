import random
import time
import tkinter as tk


class Object:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name


class Objective(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '*')

    def __str__(self):
        return f'Objective: {self.x} {self.y}'


class Obstacle(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '□')

    def __str__(self):
        return f'Obstacle: {self.x} {self.y}'

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
                seen.append(obj.name)
                break
            if obj is not None: seen.append(obj.name)

        if not seen:
            return None
        return seen


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
        # Weights: 4 inputs (Up, Down, Left, Right) -> 4 outputs (Up, Down, Left, Right)
        self.weights = {
            'up': [random.uniform(-0.1, 0.1) for _ in range(4)],
            'down': [random.uniform(-0.1, 0.1) for _ in range(4)],
            'left': [random.uniform(-0.1, 0.1) for _ in range(4)],
            'right': [random.uniform(-0.1, 0.1) for _ in range(4)]
        }
        self.path = []
        self.last_action_name = None
        self.last_input = None
        self.learning_rate = 0.1

    def process_observation(self, obs):
        # Input vector: [up_val, down_val, left_val, right_val]
        # Val: 1 if Objective, -1 if Obstacle/Wall, 0 if Empty
        mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        input_vec = [0.0] * 4
        
        if not obs:
            return input_vec

        for direction, items in obs.items():
            idx = mapping.get(direction)
            if idx is not None:
                val = 0.0
                if items:
                    closest = items[0]
                    if closest == '*': # Objective
                        val = 1.0
                    elif closest == '□' or closest != '*': # Obstacle/Agent
                        val = -1.0
                input_vec[idx] = val
        return input_vec

    def age(self) -> Accao:
        if not self.ultima_observacao:
            d = random.choice(["up", "down", "left", "right"])
            self.last_action_name = d
            self.last_input = [0.0]*4
            return Move(d)

        inputs = self.process_observation(self.ultima_observacao)
        self.last_input = inputs
        
        actions = ['up', 'down', 'left', 'right']
        best_score = -float('inf')
        best_action = random.choice(actions)
        
        for action in actions:
            w = self.weights[action]
            score = sum(i * w_i for i, w_i in zip(inputs, w))
            if score > best_score:
                best_score = score
                best_action = action
        
        # Epsilon-greedy
        if random.random() < 0.1:
            best_action = random.choice(actions)

        self.last_action_name = best_action
        return Move(best_action)

    def avalicaoEstadoAtual(self, recompensa: int):
        if self.last_action_name and self.last_input:
            w = self.weights[self.last_action_name]
            new_w = []
            for i, val in enumerate(w):
                new_w.append(val + self.learning_rate * recompensa * self.last_input[i])
            self.weights[self.last_action_name] = new_w

    def executar(self, ambiente):
        observacao = ambiente.observacaoPara(self)
        self.observacao(observacao)
        accao = self.age()
        
        target_x = self.x + accao.modifier[0]
        target_y = self.y + accao.modifier[1]
        
        if target_x < 0 or target_x >= ambiente.size or target_y < 0 or target_y >= ambiente.size:
             reward = -10
             result = False
        else:
            target_obj = ambiente.objects.get((target_x, target_y))
            if isinstance(target_obj, Objective):
                reward = 100
                print(f"{self.name} reached Objective!")
                result = False
            elif target_obj is not None:
                reward = -10
                result = False
            else:
                reward = -1
                result = ambiente.agir(accao, self)
        
        if result:
            self.path.append((self.x, self.y))
            
        self.avalicaoEstadoAtual(reward)


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

    # O mesmo que no main.py
    def executa(self):
        self.ambiente.atualizacao()
        for agent in self.listaAgentes:
            agent.executar(self.ambiente)


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
                        # items é uma lista de strings, ex: ['□', 'A']
                        self.vision_text.insert(tk.END, f"{', '.join(items)}\n")
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
            self.simulador.executa()
            self.update_gui()
            self.master.after(1000, self.run_step)  # Loop

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
    agent1 = NeuralAgent(0, 0, "N")
    agent1.instala(CircularSensor(3))
    la.append(agent1)
    agent2 = Agent(0, 1, "D")
    agent2.instala(CircularSensor(3))
    la.append(agent2)

    amb = Ambiente(10)
    amb.add_object(Obstacle(2, 2))
    amb.add_object(Obstacle(3, 2))
    amb.add_object(Obstacle(4, 2))
    amb.add_object(Objective(5, 5))
    
    simulador = Simulador(la, amb)

    root = tk.Tk()
    gui = SimulationGUI(root, simulador)
    root.mainloop()