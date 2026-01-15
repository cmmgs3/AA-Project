import json
import os
import tkinter as tk

from gui import SimulationGUI
from simulator import Simulador
from world import Ambiente, Obstacle, Farol, Objective
from agents import NeuralAgent
from helpers import CircularSensor, FarolSensor


if __name__ == "__main__":
    la = []

    ENV_SCENARIOS = [
        # Scenario 0: Original
        {
            "obstacles": [
                (2, 2), (3, 2), (4, 2),
                (4, 4), (4, 5),
                (6, 2), (7, 2), (8, 2),
                (6, 4), (6, 5),
                (7, 8), (6, 8), (5, 8), (4, 8), (3, 8),
                (8, 7), (2, 7), (9, 7)
            ],
            "objective": (8, 8),
            "start_pos": (0, 0),
            "farol": True
        },
        {
            "obstacles": [
                (2, 2), (3, 2), (4, 2),
                (4, 4), (4, 5),
                (6, 2), (7, 2), (8, 2),
                (6, 4), (6, 5),
                (7, 8), (6, 8), (5, 8), (4, 8), (3, 8),
                (8, 7), (2, 7), (7, 9)
            ],
            "objective": (8, 8),
            "start_pos": (0, 0),
            "farol": False
        },
        {
            "obstacles": [
                (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
                (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9),
                (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8),

                (8, 4), (8, 5), (8, 6),

                (2, 2), (3, 2), (4, 2),
                (7, 2), (8, 2),

                (3, 3),

                (6, 1), (6, 2), (6, 3), (6, 4),

                (5, 7), (6, 7), (7, 7)
            ],
            "objective": (8, 7),
            "start_pos": (1, 2),
            "farol": False
        },

        {
            "obstacles": [
                (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
                (1, 2), (8, 2),
                (1, 3), (3, 3), (4, 3), (5, 3), (8, 3),
                (1, 4), (5, 4), (8, 4),
                (1, 5), (3, 5), (4, 5), (5, 5), (8, 5),
                (1, 6), (8, 6),
                (1, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
                (1, 8), (8, 8),(9,1)
            ],
            "objective": (9, 0),
            "start_pos": (0, 9),
            "farol": True
        },
        {
            "obstacles": [
                (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
                (1, 2), (8, 2),
                (1, 3), (3, 3), (4, 3), (5, 3), (8, 3),
                (1, 4), (5, 4), (8, 4),
                (1, 5), (3, 5), (4, 5), (5, 5), (8, 5),
                (1, 6), (8, 6),
                (1, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
                (1, 8), (8, 8), (0,1)
            ],
            "objective": (9, 0),
            "start_pos": (0, 9),
            "farol": False
        },

        {
            "obstacles": [
                (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                (2, 3), (7, 3),
                (2, 4), (4, 4), (5, 4), (7, 4),
                (2, 5), (4, 5), (5, 5), (7, 5),
                (2, 6), (7, 6),
                (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)
            ],
            "objective": (9, 5),
            "start_pos": (0, 4),
            "farol": True
        }

    ]


    def create_environment(scenario):
        amb = Ambiente(10)
        for obs_pos in scenario["obstacles"]:
            amb.add_object(Obstacle(obs_pos[0], obs_pos[1]))

        isfarol = scenario["farol"]
        start_x, start_y = scenario["start_pos"]
        agent = NeuralAgent(start_x, start_y, "N")

        obj_pos = scenario["objective"]
        # If map has farol, loads FarolSensor, else loads CircularSensor
        if isfarol:
            farol = Farol(obj_pos[0], obj_pos[1])
            amb.add_object(farol)
            agent.instala(CircularSensor(1))
            agent.instala(FarolSensor(farol))
        else:
            amb.add_object(Objective(obj_pos[0], obj_pos[1]))
            agent.instala(CircularSensor(3))

        la.append(agent)
        return amb


    amb = create_environment(ENV_SCENARIOS[1])

    simulador = Simulador(la, amb)

    root = tk.Tk()

    # Try to load best weights if available
    if os.path.exists("best_weights.json"):
        try:
            with open("best_weights.json", "r") as f:
                weights = json.load(f)
            for a in la:
                a.set_weights(weights)
            print("Loaded best weights!")
        except Exception as e:
            print(f"Could not load weights: {e}")

    gui = SimulationGUI(root, simulador)
    root.mainloop()

