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
        target_pos = (x, y)

        if current_pos == target_pos:
            return False

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
