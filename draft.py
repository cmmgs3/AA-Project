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

class Agent(Object):
    def __init__(self, x, y, name):
        self.sensor = None
        self.ultima_observacao = None
        super().__init__(x, y, name)

    def observacao(self, obs):
        self.ultima_observacao = obs

    def printObservacao(self):
        if self.ultima_observacao is not None:
            print(self.name + " observacao")
            for key, value in self.ultima_observacao.items():
                print(key + ": " + str(value))

    def avalicaoEstadoAtual(self, recompensa):
        pass

    def instala(self, sensor):
        self.sensor = sensor

    def comunica(self, de_agente):
        pass

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
            obj = ambiente.objects.get((x,y))
            if obj is not None:
                #TODO maybe distancia?
                seen.append(obj.name)

        if not seen:
            return None
        return seen

class Accao:
    def act(self, agente, ambiente):
        pass

class Move(Accao):
    def __init__(self, direction):
        if direction != "right" and direction != "left" and direction != "up" and direction != "down":
            print("Direction must be 'right', 'left', 'up' or 'down'")
            return

        if direction == "up" :
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
            print("Can't move outside of map")
            return False
        if ambiente.objects.get(target_location) is not None:
            print("Place already occupied by " + ambiente.objects.get(target_location).name)
            return False

        ambiente.objects.pop(current_location)
        agente.x, agente.y = target_location
        ambiente.objects.update({target_location: agente})
        return True
    

class Ambiente:
    def __init__(self, size):
        self.objects = {}
        self.size = size

    def add_object(self, obj):
        target_pos = (obj.x, obj.y)
        if self.objects.get(target_pos) is None:
            self.objects[target_pos] = obj

    def observacaoPara(self, agente):
        if isinstance(agente, Agent):
            if agente.sensor is not None:
                return agente.sensor.sense(agente, self)
            else:
                print("Agent: " + agente.name + " nao tem sensor")
        return None

    def atualizacao(self):
        pass

    def agir(self, accao, agente):
        if isinstance(accao, Accao) and isinstance(agente, Agent):
            accao.act(agente, self)

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
    def __init__(self, listaAgente, ambiente):
        self.listaAgentes = listaAgente
        self.ambiente = ambiente
        for agent in la:
            self.ambiente.add_object(agent)


    def listarAgentes(self):
        for agent in self.listaAgentes:
            print(agent)

    # O mesmo que no main.py
    def executa(self):
        directions = ["up", "down", "left", "right"]
        # 2. Ação
        for agent in self.listaAgentes:
            move = Move(random.choice(directions))
            self.ambiente.agir(move, agent)
        # 1. Percepção
        for agent in self.listaAgentes:
            observacao = self.ambiente.observacaoPara(agent)
            agent.observacao(observacao)
            # agent.printObservacao() # Opcional, suja o console

        self.ambiente.atualizacao()

class SimulationGUI:
    def __init__(self, master, simulador):
        self.master = master
        self.simulador = simulador
        self.ambiente = simulador.ambiente
        self.running = False

        master.title("Simulação Multi-Agente")
        master.configure(bg='#f0f0f0')

        #Layout Principa
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

            obs = agent.ultima_observacao
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
    agent1 = Agent(0, 1, "A")
    agent1.instala(CircularSensor(2))
    agent2 = Agent(0, 2, "B")
    agent2.instala(CircularSensor(1))
    la.append(agent1)
    la.append(agent2)
    amb = Ambiente(9)
    amb.add_object(Obstacle(0,3))
    amb.add_object(Obstacle(1,3))
    amb.add_object(Obstacle(2,3))
    amb.add_object(Obstacle(3,3))
    simulador = Simulador(la, amb)

    root = tk.Tk()
    gui = SimulationGUI(root, simulador)
    root.mainloop()
    #simulador.executa()