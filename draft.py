import random
import time


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

    def executa(self):
        directions = ["up", "down", "left", "right"]
        while True:
            self.ambiente.print()
            for agent in self.listaAgentes:
                observacao = self.ambiente.observacaoPara(agent)
                agent.observacao(observacao)
                agent.printObservacao()
                move = Move(random.choice(directions))
                self.ambiente.agir(move, agent)
            self.ambiente.atualizacao()
            time.sleep(1)

if __name__ == "__main__":
    la = []
    agent1 = Agent(0, 1, "A")
    agent1.instala(CircularSensor(2))
    agent2 = Agent(0, 2, "B")
    la.append(agent1)
    la.append(agent2)
    amb = Ambiente(9)
    amb.add_object(Obstacle(0,3))
    amb.add_object(Obstacle(1,3))
    amb.add_object(Obstacle(2,3))
    amb.add_object(Obstacle(3,3))
    simulador = Simulador(la, amb)
    simulador.executa()