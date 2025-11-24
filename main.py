import random
import time

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
    def move_random(self, w):
        size = w.size
        directions =["up", "down", "left", "right"]
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


    def __str__(self):
        return f'Agent: {self.x} {self.y} {self.name}'

class Objective(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '*')
    def __str__(self):
        return f'Objective: {self.x} {self.y}'
    def intersect(self):
        return True

class Obstacle(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '⬛')
    def __str__(self):
        return f'Obstacle: {self.x} {self.y}'
    def intersect(self):
        return False


class World:
    def __init__(self, size):
        self.objects = {}
        self.size = size
        self.agents = []
    def add_object(self, o):
        target_pos = (o.x, o.y)
        if self.objects.get(target_pos) is None:
            self.objects[target_pos] = []
            self.objects[target_pos].append(o)
        if self.objects.get(target_pos) is not None:
            self.objects[target_pos].append(o)
        if isinstance(o, Agent) and self.agents.count(o) == 0:
            self.agents.append(o)

    def get_at(self, x, y):
        return self.objects.get((x, y))
    def move(self, agent, x, y):
        current_pos = (agent.x, agent.y)
        target_pos = (x,y)

        if current_pos == target_pos:
            return False

        objs_at_targ = self.objects.get(target_pos)
        can_move = True
        if objs_at_targ is not None:
            for obj in objs_at_targ:
                can_move = can_move and obj.intersect

        if can_move:
            objs_at_current_pos = self.objects.get(current_pos)
            objs_at_current_pos.remove(agent)
            agent.x = x
            agent.y = y
            self.add_object(agent)
            return True
        return False

    def display(self):
        matrix = [["." for x in range(self.size)] for y in range(self.size)]
        for objs in self.objects.values():
            for obj in objs:
                if matrix[obj.x][obj.y] == ".":
                    matrix[obj.x][obj.y] = obj.name
        for row in matrix:
            print(' '.join(row))
    def update(self):
        for agent in self.agents:
            agent.move_random(self)


if __name__ == '__main__':
    print("Hello World")
    world = World(3)
    world.add_object(Agent(0, 1, "A"))
    world.add_object(Agent(0, 2, "B"))
    while True:
        world.update()
        world.display()
        time.sleep(1)
        print('\n')

