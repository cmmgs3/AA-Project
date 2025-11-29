import random
import json
import copy
import math
from main import NeuralAgent, CircularSensor, Ambiente, Obstacle, Objective, Simulador

# Configuration
POPULATION_SIZE = 50
GENERATIONS = 100
SIMULATION_STEPS = 50
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
ENV_SIZE = 10

# Define multiple environment scenarios
ENV_SCENARIOS = [
    # Scenario 0: Original
    {
        "obstacles": [(2, 2), (3, 2), (4, 2)],
        "objective": (5, 5),
        "start_pos": (0, 0)
    },
    # Scenario 1: Wall
    {
        "obstacles": [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)],
        "objective": (8, 2),
        "start_pos": (2, 2)
    },
    # Scenario 2: Corner to Corner
    {
        "obstacles": [(3, 3), (4, 4), (5, 5), (6, 6)],
        "objective": (9, 9),
        "start_pos": (0, 0)
    },
    # Scenario 3: Trap (U-shape)
    {
        "obstacles": [(3, 3), (4, 3), (5, 3), (3, 4), (5, 4), (3, 5), (4, 5), (5, 5)],
        "objective": (4, 4), # Inside the U
        "start_pos": (0, 0)
    },
    # Scenario 4: Open Field
    {
        "obstacles": [],
        "objective": (7, 7),
        "start_pos": (1, 1)
    }
]

def create_environment(scenario):
    amb = Ambiente(ENV_SIZE)
    for obs_pos in scenario["obstacles"]:
        amb.add_object(Obstacle(obs_pos[0], obs_pos[1]))
    
    obj_pos = scenario["objective"]
    amb.add_object(Objective(obj_pos[0], obj_pos[1]))
    return amb

def run_episode(weights, scenario):
    start_x, start_y = scenario["start_pos"]
    agent = NeuralAgent(start_x, start_y, "N")
    agent.instala(CircularSensor(3))
    agent.set_weights(weights)
    
    amb = create_environment(scenario)
    sim = Simulador([agent], amb)
    
    total_reward = 0
    
    obj_x, obj_y = scenario["objective"]
    
    for _ in range(SIMULATION_STEPS):
        sim.executa()
        total_reward -= 1 # Step penalty
        
        # Check adjacency to Objective
        dist = abs(agent.x - obj_x) + abs(agent.y - obj_y)
        
        if dist <= 1:
            total_reward += 100
            break 
            
    return total_reward

def create_random_weights():
    # 9 inputs: Up, Down, Left, Right, Bias, LastUp, LastDown, LastLeft, LastRight
    return {
        'up': [random.uniform(-1, 1) for _ in range(9)],
        'down': [random.uniform(-1, 1) for _ in range(9)],
        'left': [random.uniform(-1, 1) for _ in range(9)],
        'right': [random.uniform(-1, 1) for _ in range(9)]
    }

def mutate(weights):
    new_weights = copy.deepcopy(weights)
    for action in new_weights:
        for i in range(len(new_weights[action])):
            if random.random() < MUTATION_RATE:
                new_weights[action][i] += random.gauss(0, MUTATION_STRENGTH)
    return new_weights

def crossover(parent1, parent2):
    child = {}
    for action in parent1:
        child[action] = []
        for i in range(len(parent1[action])):
            if random.random() < 0.5:
                child[action].append(parent1[action][i])
            else:
                child[action].append(parent2[action][i])
    return child

def main():
    print(f"Starting Multi-Environment Training: Pop={POPULATION_SIZE}, Gens={GENERATIONS}, Scenarios={len(ENV_SCENARIOS)}")
    
    population = [create_random_weights() for _ in range(POPULATION_SIZE)]
    best_overall_weights = None
    best_overall_score = -float('inf')
    
    for gen in range(GENERATIONS):
        scores = []
        for weights in population:
            # Evaluate on ALL scenarios and average the score
            total_scenario_score = 0
            for scenario in ENV_SCENARIOS:
                total_scenario_score += run_episode(weights, scenario)
            
            avg_score = total_scenario_score / len(ENV_SCENARIOS)
            scores.append((avg_score, weights))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        best_score = scores[0][0]
        avg_gen_score = sum(s[0] for s in scores) / len(scores)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_weights = scores[0][1]
            
        print(f"Gen {gen+1}: Best={best_score:.2f}, Avg={avg_gen_score:.2f}")
        
        # Selection: Keep top 20%
        survivors = [s[1] for s in scores[:int(POPULATION_SIZE * 0.2)]]
        
        new_population = []
        new_population.extend(survivors) # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population
        
    print(f"Training Complete. Best Score: {best_overall_score}")
    
    if best_overall_weights:
        with open("best_weights.json", "w") as f:
            json.dump(best_overall_weights, f)
        print("Saved best weights to best_weights.json")

if __name__ == "__main__":
    main()
