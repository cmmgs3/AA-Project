import random
import time
import tkinter as tk

# Constantes para a simulação
SIMULATION_DELAY_MS = 1000  # Meio segundo entre os passos
VISION_RANGE = 4  # Alcance de visão para o agente


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
        super().__init__(x, y, name)
        self.path = [(x, y)]  # Caminho percorrido

    def move_random(self, w):
        size = w.size
        directions = ["up", "down", "left", "right"]
        # Tenta no máximo 10 vezes para evitar loops infinitos se o agente estiver preso
        for _ in range(10):
            move = random.choice(directions)
            if move == "up" and self.y > 0:
                if w.move(self, self.x, self.y - 1):
                    break
            elif move == "down" and self.y < size - 1:
                if w.move(self, self.x, self.y + 1):
                    break
            elif move == "left" and self.x > 0:
                if w.move(self, self.x - 1, self.y):
                    break
            elif move == "right" and self.x < size - 1:
                if w.move(self, self.x + 1, self.y):
                    break

    def watch_direction(self, w, direction):
        # O agente olha até VISION_RANGE células
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

        # Percorre de 1 até o alcance máximo de visão
        for i in range(1, VISION_RANGE + 1):
            x, y = (self.x + (modifier[0] * i), self.y + (modifier[1] * i))

            # Verifica se está dentro dos limites do mundo
            if 0 <= x < w.size and 0 <= y < w.size:
                obj = w.get_at(x, y)
                if obj is not None:
                    # Retorna o nome do objeto e a distância (i)
                    seen.append((obj.name, i))
            else:
                # Sai se atingir o limite do mundo
                break

        return seen

    def watch_environment(self, w):
        up_list = self.watch_direction(w, "up")
        down_list = self.watch_direction(w, "down")
        left_list = self.watch_direction(w, "left")
        right_list = self.watch_direction(w, "right")
        return {"up": up_list, "down": down_list, "left": left_list, "right": right_list}

    def printpath(self):
        print(f"--- Caminho de {self.name} ---")
        for i, pos in enumerate(self.path):
            print(f"Passo {i}: {pos}\n")
        print("--- Fim do Caminho ---")


class World:
    def __init__(self, size):
        self.objects = {}
        self.size = size
        self.agents = []

    def add_object(self, obj):
        target_pos = (obj.x, obj.y)
        if self.objects.get(target_pos) is None:
            self.objects[target_pos] = obj
        if isinstance(obj, Agent) and self.agents.count(obj) == 0:
            self.agents.append(obj)

    def get_at(self, x, y):
        # Lida com coordenadas fora do limite
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.objects.get((x, y))
        return None

    def move(self, agent, x, y):
        current_pos = (agent.x, agent.y)
        target_pos = (x, y)
        if current_pos == target_pos:
            return False

        # Verifica se a posição alvo está ocupada
        target_obj = self.objects.get(target_pos)
        if target_obj is not None:
            # Agentes podem interagir com Objetivos, mas não Obstruções
            if isinstance(target_obj, Obstacle):
                # print(f"Lugar ({x},{y}) já ocupado por Obstáculo. Movimento bloqueado.")
                return False
            elif isinstance(target_obj, Objective):
                # print(f"Agente {agent.name} alcançou Objetivo em ({x},{y})!")
                # Permite que o agente se mova para o objetivo (opcionalmente remove o objetivo)
                # Neste exemplo, apenas permite o movimento.
                pass

                # Move o agente
        self.objects.pop(current_pos)
        agent.x, agent.y = target_pos
        self.objects.update({target_pos: agent})
        agent.path.append(target_pos)

        return True

    def update(self):
        """
        Executa o passo de simulação para todos os agentes, coleta a visão
        e retorna-a para a GUI.
        """
        all_visions = {}
        for agent in self.agents:
            # 1. Agente detecta o ambiente
            vision = agent.watch_environment(self)
            # Armazena a visão para ser exibida na GUI
            all_visions[agent.name] = vision

            # 2. Agente se move
            agent.move_random(self)

        # RETORNA A VISÃO para a GUI, resolvendo o bug.
        return all_visions


class SimulationGUI:
    def __init__(self, master, world):
        self.master = master
        self.world = world
        self.running = False

        master.title("Simulação Agentes/Mundo (Tk)")
        master.configure(bg='#f0f0f0')

        # --- Configuração da Grade Principal ---

        self.main_frame = tk.Frame(master, padx=15, pady=15, bg='#f0f0f0')
        self.main_frame.pack(fill="both", expand=True)

        # Frame para o Mapa (Esquerda)
        self.map_frame = tk.Frame(self.main_frame, padx=10, pady=10, bg='#ffffff', relief=tk.RAISED, borderwidth=2)
        self.map_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Frame para os Controles e Visão (Direita)
        self.info_frame = tk.Frame(self.main_frame, padx=10, pady=10, bg='#e0e0e0', relief=tk.RIDGE, borderwidth=1)
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # --- Mapa (Grid de tk.Labels) ---

        self.cells = []
        for r in range(self.world.size):
            row_cells = []
            for c in range(self.world.size):
                cell = tk.Label(self.map_frame, text=".", width=3, height=1,
                                borderwidth=1, relief="ridge", font=('Inter', 14, 'bold'), fg='#333333', bg='#f0f0f0')
                cell.grid(row=r, column=c, padx=1, pady=1)
                row_cells.append(cell)
            self.cells.append(row_cells)

        # --- Controles ---
        self.control_frame = tk.Frame(self.info_frame, bg='#e0e0e0')
        self.control_frame.pack(pady=10, fill='x')

        self.start_button = tk.Button(self.control_frame, text="▶ Iniciar Simulação", command=self.start_simulation,
                                      bg='#4CAF50', fg='white', font=('Inter', 10, 'bold'), relief=tk.RAISED)
        self.start_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.stop_button = tk.Button(self.control_frame, text="■ Parar Simulação", command=self.stop_simulation,
                                     state=tk.DISABLED, bg='#F44336', fg='white', font=('Inter', 10, 'bold'),
                                     relief=tk.RAISED)
        self.stop_button.pack(side=tk.LEFT, expand=True, padx=5)

        # --- Visão dos Agentes ---

        tk.Label(self.info_frame, text="👁 Visão dos Agentes (Próximo [Distância])",
                 font=('Inter', 12, 'bold'), bg='#e0e0e0', fg='#333333').pack(pady=(15, 5))

        self.vision_text = tk.Text(self.info_frame, height=15, width=40, state=tk.DISABLED,
                                   font=('Consolas', 10), bg='#ffffff', fg='#000000', relief=tk.FLAT)
        self.vision_text.pack(pady=5, fill='both', expand=True)

        self.update_map()

    def update_map(self):
        """Atualiza a grade do mapa na GUI."""
        for r in range(self.world.size):
            for c in range(self.world.size):
                obj = self.world.get_at(c, r)

                # Definição de cores e texto
                text = "."
                bg_color = '#ffffff'
                fg_color = '#cccccc'

                if obj:
                    text = obj.name
                    if isinstance(obj, Agent):
                        bg_color = '#4DD0E1'  # Ciano Claro
                        fg_color = '#000000'
                    elif isinstance(obj, Obstacle):
                        bg_color = '#757575'  # Cinza Escuro
                        fg_color = '#ffffff'
                    elif isinstance(obj, Objective):
                        bg_color = '#8BC34A'  # Verde Limão
                        fg_color = '#000000'

                self.cells[r][c].config(text=text, bg=bg_color, fg=fg_color, relief='raised' if obj else 'ridge')

    def update_vision_text(self, all_visions):
        """Atualiza a área de texto com a visão dos agentes."""
        self.vision_text.config(state=tk.NORMAL)
        self.vision_text.delete(1.0, tk.END)

        if not all_visions:
            self.vision_text.insert(tk.END, "Nenhuma visão de agente disponível.")
            self.vision_text.config(state=tk.DISABLED)
            return

        for agent_name, vision in all_visions.items():
            self.vision_text.insert(tk.END, f"--- Agente {agent_name} ---\n", 'agent_header')

            for direction, items in vision.items():
                self.vision_text.insert(tk.END, f"  {direction.capitalize()}: ")
                if items:
                    # items é uma lista de (name, distance)
                    items_str = ", ".join([f"{name} [{dist}]" for name, dist in items])
                    self.vision_text.insert(tk.END, items_str + "\n")
                else:
                    self.vision_text.insert(tk.END, "Nada visto\n", 'none_seen')
            self.vision_text.insert(tk.END, "\n")

        self.vision_text.config(state=tk.DISABLED)

        # Adiciona tags de estilo para melhor visualização
        self.vision_text.tag_config('agent_header', font=('Inter', 10, 'bold'), foreground='#1E88E5')
        self.vision_text.tag_config('none_seen', foreground='#9E9E9E')

    def simulation_step(self):
        """Executa um passo da simulação e atualiza a GUI."""
        if not self.running:
            return

        # Agora self.world.update() retorna all_visions, corrigindo o erro
        all_visions = self.world.update()

        # Atualiza a visualização na tela
        self.update_map()
        self.update_vision_text(all_visions)

        # Agendar o próximo passo após o tempo definido
        self.master.after(SIMULATION_DELAY_MS, self.simulation_step)

    def start_simulation(self):
        """Inicia o loop da simulação."""
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.simulation_step()

    def stop_simulation(self):
        """Para o loop da simulação."""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)


if __name__ == '__main__':
    # Configuração do Mundo
    world = World(9)
    world.add_object(Agent(0, 1, "A"))
    world.add_object(Agent(0, 2, "B"))

    # Adiciona uma parede de obstáculos
    for i in range(1, 8):
        world.add_object(Obstacle(i, 3))

    world.add_object(Objective(8, 8))
    world.add_object(Objective(4, 4))

    # Inicia a GUI
    root = tk.Tk()
    app = SimulationGUI(root, world)
    root.mainloop()