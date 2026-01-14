class Object:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name


class Objective(Object):
    def __init__(self, x, y):
        super().__init__(x, y, '*')

    def checkWin(self, ambiente) -> bool:
        up = ambiente.objects.get((self.x, self.y - 1))
        down = ambiente.objects.get((self.x, self.y + 1))
        left = ambiente.objects.get((self.x - 1, self.y))
        right = ambiente.objects.get((self.x + 1, self.y))
        # Use duck-typing: an Agent has a 'sensor' attribute, obstacles and other objects don't
        if (hasattr(up, 'sensor') or hasattr(down, 'sensor') or
                hasattr(left, 'sensor') or hasattr(right, 'sensor')):
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


class Ambiente:
    def __init__(self, size: int):
        self.objects = {}
        self.size = size

    def add_object(self, obj: Object):
        target_pos = (obj.x, obj.y)
        if self.objects.get(target_pos) is None:
            self.objects[target_pos] = obj

    def observacaoPara(self, agente) -> dict | None:
        # Avoid importing Agent here; use duck-typing so other modules don't create circular imports
        if hasattr(agente, 'sensor') and agente.sensor:
            obs = {}
            for sensor in agente.sensor:
                obs.update(sensor.sense(agente, self))
            return obs
        return None

    def atualizacao(self):
        pass

    def agir(self, accao, agente):
        # Use duck-typing instead of type checks to avoid circular imports
        if hasattr(accao, 'act') and hasattr(agente, 'x') and hasattr(agente, 'y'):
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