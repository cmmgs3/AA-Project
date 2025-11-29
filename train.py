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

def create_environment():
    amb = Ambiente(ENV_SIZE)
    # Same setup as main.py for consistency
    amb.add_object(Obstacle(2, 2))
    amb.add_object(Obstacle(3, 2))
    amb.add_object(Obstacle(4, 2))
    amb.add_object(Objective(5, 5))
    return amb

def run_episode(weights):
    agent = NeuralAgent(0, 0, "N")
    agent.instala(CircularSensor(3))
    agent.set_weights(weights)
    
    amb = create_environment()
    sim = Simulador([agent], amb)
    
    total_reward = 0
    
    # Objective position
    obj_x, obj_y = 5, 5
    
    for _ in range(SIMULATION_STEPS):
        sim.executa()
        total_reward -= 1 # Step penalty (optimizes for shortest path)
        
        # Check adjacency to Objective
        # Manhattan distance == 1 means adjacent (or 0 if on top, though main.py blocks that)
        dist = abs(agent.x - obj_x) + abs(agent.y - obj_y)
        
        if dist <= 1:
            total_reward += 100
            # End simulation for this agent as requested
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
    print(f"Starting Shortest Path Training (Sparse Reward, Stop on Adj): Pop={POPULATION_SIZE}, Gens={GENERATIONS}")
    
    population = [create_random_weights() for _ in range(POPULATION_SIZE)]
    best_overall_weights = None
    best_overall_score = -float('inf')
    
    for gen in range(GENERATIONS):
        scores = []
        for weights in population:
            score = run_episode(weights)
            scores.append((score, weights))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        best_score = scores[0][0]
        avg_score = sum(s[0] for s in scores) / len(scores)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_weights = scores[0][1]
            
        print(f"Gen {gen+1}: Best={best_score}, Avg={avg_score:.2f}")
        
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
