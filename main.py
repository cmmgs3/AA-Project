import random
import time

class Agent:
    def __init__(self, x,y, name):
        self.x = x
        self.y = y
        self.name = name
    def __str__(self):
        return f'{self.x} {self.y} {self.name}'
    def move_random(self, size):
        directions =["up", "down", "left", "right"]
        while True:
            move = random.choice(directions)
            if move == "up" and self.y > 0:
                self.y -= 1
                break
            elif move == "down" and self.y < size - 1:
                self.y += 1
                break
            elif move == "left" and self.x > 0:
                self.x -= 1
                break
            elif move == "right" and self.x < size - 1:
                self.x += 1
                break


class Objective:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.name = '*'
    def __str__(self):
        return f'{self.x} {self.y}'


class World:
    def __init__(self, size):
        self.agents = []
        self.size = size
    def add_agent(self, agent):
        self.agents.append(agent)

    def display(self):
        matrix = [["." for x in range(self.size)] for y in range(self.size)]
        for agent in self.agents:
            x, y = agent.x, agent.y
            matrix[y][x] = agent.name
        for row in matrix:
            print(' '.join(row))
    def update(self):
        for agent in self.agents:
            agent.move_random(self.size)

if __name__ == '__main__':
    print("Hello World")
    world = World(3)
    world.add_agent(Agent(0, 1, "A"))
    world.add_agent(Agent(0, 2, "B"))
    while True:
        world.update()
        world.display()
        time.sleep(1)
        print('\n')

