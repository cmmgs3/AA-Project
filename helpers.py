from world import Obstacle, Farol


class Accao:
    def act(self, agente, ambiente):
        pass

class Move(Accao):
    def __init__(self, direction):
        if direction not in ("up", "down", "left", "right"):
            raise ValueError(f"Invalid move direction: {direction}")

        if direction == "up":
            self.modifier = (0, -1)
        elif direction == "down":
            self.modifier = (0, 1)
        elif direction == "right":
            self.modifier = (1, 0)
        elif direction == "left":
            self.modifier = (-1, 0)

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
                seen.append((obj.name, i))
                break
            if obj is not None: seen.append((obj.name, i))

        if not seen:
            return None
        return seen


class FarolSensor(Sensor):
    def __init__(self, farol: Farol):
        self.farol = farol

    def sense(self, agente, ambiente):
        return {"farol_coord" : (self.farol.x, self.farol.y)}