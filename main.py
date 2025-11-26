import random
import time
import tkinter as tk

class Object:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __str__(self):
        return f'{self.x} {self.y} {self.name}'

    def intersect(self):
        return True


class Agent(Object):
    def __init__(self, x, y, name):
        super().__init__(x, y, name)

    def watch_direction(self, w, direction):
        vision_range = 5
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

        for i in range(1, vision_range):
            x, y = (self.x + (modifier[0] * i), self.y + (modifier[1] * i))
            objs = w.get_at(x, y)
            if objs is not None:
                for obj in objs:
                    seen.append((obj.name, i))

        if not seen:
            return None
        return seen


    def watch_environment(self, w):
        up_list = self.watch_direction(w, "up")
        down_list = self.watch_direction(w, "down")
        left_list = self.watch_direction(w, "left")
        right_list = self.watch_direction(w, "right")
        return {"up": up_list, "down": down_list, "left": left_list, "right": right_list}

    def move_random(self, w):
        size = w.size
        directions = ["up", "down", "left", "right"]
        while True:
            move = random.choice(directions)
            if move == "up" and self.y > 0:
                w.move(self, self.x, self.y - 1)
                break
            elif move == "down" and self.y < size - 1:
                w.move(self, self.x, self.y + 1)
                break
            elif move == "left" and self.x > 0:
                w.move(self, self.x - 1, self.y)
                break
            elif move == "right" and self.x < size - 1:
                w.move(self, self.x + 1, self.y)
                break

    def printvision(self, vision, direction):
        if vision is not None:
            items = vision.get(direction)
            print(direction + ":")
            if items is not None:
                for item in items:
                    name = item[0]
                    distance = item[1]
                    print(name +  " at%2d spaces"%distance)




    def __str__(self):
        return f'Agent: {self.x} {self.y} {self.name}'


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

    def age(self):
        directions = ["up", "down", "left", "right"]
        return Move(random.choice(directions))

    def avalicaoEstadoAtual(self, recompensa):
        pass

    def instala(self, sensor):
        self.sensor = sensor

    def comunica(self, de_agente):
        pass

    def executar(self, ambiente):
        observacao = ambiente.observacaoPara(self)
        self.observacao(observacao)
        accao = self.age()
        result = ambiente.agir(accao, self)
        self.avalicaoEstadoAtual()

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
            return False
        if ambiente.objects.get(target_location) is not None:
            return False

        ambiente.objects.pop(current_location)
        agente.x, agente.y = target_location
        ambiente.objects.update({target_location: agente})
        return True
    

class Ambiente:
    def __init__(self, size):
        self.objects = {}
        self.size = size

        objs_at_targ = self.objects.get(target_pos)
        can_move = True
        if objs_at_targ is not None:
            for obj in objs_at_targ:
                can_move = can_move and obj.intersect

        if can_move:
            self.objects.get(current_pos).remove(agent)
            agent.x = x
            agent.y = y
            self.add_object(agent)
            return True
        return False

    def display(self):
        matrix = [["." for y in range(self.size)] for x in range(self.size)]
        for objs in self.objects.values():
            for obj in objs:
                if matrix[obj.y][obj.x] == ".":
                    matrix[obj.y][obj.x] = obj.name
        for row in matrix:
            print(' '.join(row))

    def update(self):
        for agent in self.agents:
            print("Agent " + agent.name + " moving")
            agent.move_random(self)
            print("Agent " + agent.name + " sensing")
            vision = agent.watch_environment(self)
            agent.printvision(vision, "up")
            agent.printvision(vision, "down")
            agent.printvision(vision, "left")
            agent.printvision(vision, "right")


if __name__ == '__main__':
    print("Hello World")
    world = World(7)
    world.add_object(Agent(0, 1, "A"))
    world.add_object(Agent(0, 2, "B"))
    while True:
        world.display()
        world.update()
        time.sleep(1)
        print('\n')
