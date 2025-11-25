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
        super().__init__(x, y, '⬛')

    def __str__(self):
        return f'Obstacle: {self.x} {self.y}'


class Agent(Object):
    def __init__(self, x, y, name):
        super().__init__(x, y, name)

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
            x, y = (self.x + (modifier[0]*i), self.y + (modifier[1]*i))
            obj = w.get_at(x, y)
            if obj is not None:
                seen.append(obj.name)

        if not seen:
            return None
        return seen

    def watch_environment(self, w):
        up_list = self.watch_direction(w, "up")
        down_list = self.watch_direction(w, "down")
        left_list = self.watch_direction(w, "left")
        right_list = self.watch_direction(w, "right")
        return {"up": up_list, "down": down_list, "left": left_list, "right": right_list}

    def printvision(self, vision, direction):
        if vision is not None:
            items = vision.get(direction)
            print(direction + ":")
            if items is not None:
                for item in items:
                    print(item)

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
        return self.objects.get((x, y))

    def move(self, agent, x, y):
        current_pos = (agent.x, agent.y)
        target_pos = (x, y)
        if current_pos == target_pos:
            return False
        if self.objects.get(target_pos) is not None:
            print("Place already occupied by " + self.objects.get(target_pos).name)
            return False

        self.objects.pop(current_pos)
        agent.x, agent.y = target_pos
        self.objects.update({target_pos: agent})

        return True

    def display(self):
        matrix = [["." for y in range(self.size)] for x in range(self.size)]
        for obj in self.objects.values():
            if matrix[obj.y][obj.x] == ".":
                matrix[obj.y][obj.x] = obj.name
        for row in matrix:
            print(' '.join(row))

    def update(self):
        for agent in self.agents:
            print("Agent " + agent.name + " sensing")
            vision = agent.watch_environment(self)
            agent.printvision(vision, "up")
            agent.printvision(vision, "down")
            agent.printvision(vision, "left")
            agent.printvision(vision, "right")
            print("Agent " + agent.name + " moving")
            agent.move_random(self)


if __name__ == '__main__':
    print("Hello World")
    world = World(9)
    world.add_object(Agent(0, 1, "A"))
    world.add_object(Agent(0, 2, "B"))
    world.add_object(Obstacle(0, 3))
    world.add_object(Obstacle(1, 3))
    world.add_object(Obstacle(2, 3))
    world.add_object(Obstacle(3, 3))
    while True:
        world.display()
        world.update()
        time.sleep(1)
        print('\n')