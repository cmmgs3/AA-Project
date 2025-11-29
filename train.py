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

# Novelty Parameters
NOVELTY_K = 15  # Number of nearest neighbors to consider
NOVELTY_WEIGHT = 0.5 # Weight of novelty in final fitness

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
    
    obj_x, obj_y = scenario["objective"]
    initial_dist = abs(start_x - obj_x) + abs(start_y - obj_y)
    min_dist = initial_dist
    
    steps_taken = 0
    reached_objective = False
    
    for _ in range(SIMULATION_STEPS):
        win = sim.executa()
        steps_taken += 1
        
        # Check current distance
        current_dist = abs(agent.x - obj_x) + abs(agent.y - obj_y)
        if current_dist < min_dist:
            min_dist = current_dist
            
        if win:
            reached_objective = True
            break
            
    # Calculate Performance Score
    # Reward for getting closer (0 to initial_dist)
    dist_improvement = initial_dist - min_dist
    
    # Base score from improvement
    performance_score = dist_improvement * 10 
    
    if reached_objective:
        performance_score += 100 # Big bonus for winning
        # Bonus for speed (less steps is better)
        performance_score += (SIMULATION_STEPS - steps_taken) * 2
    else:
        # Penalty for time wasted if not reached? Maybe not needed if we reward improvement.
        pass

    return {
        "performance": performance_score,
        "final_pos": (agent.x, agent.y),
        "path": agent.path, # Assuming agent stores path
        "reached": reached_objective
    }

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

def calculate_novelty(population_results):
    """
    Calculates novelty score for each agent based on final position sparsity.
    population_results: list of result dicts from run_episode
    Returns: list of novelty scores corresponding to the input list
    """
    novelty_scores = []
    
    # Extract all final positions
    # We aggregate final positions across ALL scenarios for a single agent? 
    # Or calculate novelty per scenario and average?
    # Let's assume population_results is a list of [ (avg_perf, weights, [final_pos_scen1, final_pos_scen2...]) ]
    # Actually, let's simplify: Novelty is based on the behavior vector.
    # Behavior vector = Concatenation of final (x,y) for all scenarios.
    
    behavior_vectors = []
    for res in population_results:
        # res is a dict containing 'behavior_vector' which is a list of coords
        behavior_vectors.append(res['behavior_vector'])
        
    for i, my_behavior in enumerate(behavior_vectors):
        distances = []
        for j, other_behavior in enumerate(behavior_vectors):
            if i == j: continue
            
            # Euclidean distance between behavior vectors
            dist = 0
            for k in range(len(my_behavior)):
                p1 = my_behavior[k]
                p2 = other_behavior[k]
                dist += (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            dist = math.sqrt(dist)
            distances.append(dist)
            
        # Sparsity = Average distance to k-nearest neighbors
        distances.sort()
        k_nn = distances[:NOVELTY_K]
        if k_nn:
            sparsity = sum(k_nn) / len(k_nn)
        else:
            sparsity = 0
        novelty_scores.append(sparsity)
        
    return novelty_scores

def main():
    print(f"Starting Multi-Environment Training with Novelty Search")
    print(f"Pop={POPULATION_SIZE}, Gens={GENERATIONS}, Scenarios={len(ENV_SCENARIOS)}")
    
    population = [create_random_weights() for _ in range(POPULATION_SIZE)]
    best_overall_weights = None
    best_overall_score = -float('inf')
    
    for gen in range(GENERATIONS):
        # 1. Evaluation Phase
        pop_results = [] # Stores {weights, performance, behavior_vector}
        
        for weights in population:
            total_perf = 0
            behavior_vector = []
            
            for scenario in ENV_SCENARIOS:
                result = run_episode(weights, scenario)
                total_perf += result["performance"]
                behavior_vector.append(result["final_pos"])
            
            avg_perf = total_perf / len(ENV_SCENARIOS)
            pop_results.append({
                "weights": weights,
                "performance": avg_perf,
                "behavior_vector": behavior_vector
            })
            
        # 2. Novelty Calculation
        novelty_scores = calculate_novelty(pop_results)
        
        # 3. Fitness Assignment & Stats
        final_scored_pop = []
        gen_best_perf = -float('inf')
        
        for i, res in enumerate(pop_results):
            novelty = novelty_scores[i]
            fitness = res["performance"] + (novelty * NOVELTY_WEIGHT)
            
            final_scored_pop.append((fitness, res["weights"], res["performance"]))
            
            if res["performance"] > gen_best_perf:
                gen_best_perf = res["performance"]
                
        # Sort by FITNESS
        final_scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = final_scored_pop[0][0]
        best_perf_in_gen = final_scored_pop[0][2] # Perf of the fittest
        
        # Check global best (based on pure performance, or fitness? Usually performance is what we care about ultimately)
        # Let's track best performance.
        if gen_best_perf > best_overall_score:
            best_overall_score = gen_best_perf
            # Find the weights associated with this best performance
            # (Note: The sorted list is by fitness, so we need to find the max perf one)
            best_perf_entry = max(final_scored_pop, key=lambda x: x[2])
            best_overall_weights = best_perf_entry[1]
            
        avg_fitness = sum(x[0] for x in final_scored_pop) / len(final_scored_pop)
        
        print(f"Gen {gen+1}: Best Fitness={best_fitness:.2f}, Best Perf={gen_best_perf:.2f}, Avg Fit={avg_fitness:.2f}")
        
        # 4. Selection
        survivors = [x[1] for x in final_scored_pop[:int(POPULATION_SIZE * 0.2)]]
        
        new_population = []
        new_population.extend(survivors) # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population
        
    print(f"Training Complete. Best Overall Performance: {best_overall_score}")
    
    if best_overall_weights:
        with open("best_weights.json", "w") as f:
            json.dump(best_overall_weights, f)
        print("Saved best weights to best_weights.json")

if __name__ == "__main__":
    main()
