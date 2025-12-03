import random
import json
import copy
import math

from main import Ambiente, Obstacle, Objective, NeuralAgent, CircularSensor, Simulador

POPULATION_SIZE = 50
GENERATIONS = 5  # Reduzi um pouco para veres resultados mais depressa
SIMULATION_STEPS = 20
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
ENV_SIZE = 10

# Novelty Parameters
NOVELTY_K = 15
NOVELTY_WEIGHT = 0.5

ENV_SCENARIOS = [
    {"obstacles": [(2, 2), (3, 2), (4, 2)], "objective": (5, 5), "start_pos": (0, 0)},
    {"obstacles": [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)], "objective": (8, 2), "start_pos": (2, 2)},
    {"obstacles": [(3, 3), (4, 4), (5, 5), (6, 6)], "objective": (9, 9), "start_pos": (0, 0)},
    {"obstacles": [(3, 3), (4, 3), (5, 3), (3, 4), (5, 4), (3, 5), (4, 5), (5, 5)], "objective": (4, 4),
     "start_pos": (0, 0)},
    {"obstacles": [], "objective": (7, 7), "start_pos": (1, 1)},
    {
            "obstacles": [
                (2, 2), (3, 2), (4, 2),
                (4, 4), (4, 5),
                (6, 2), (7, 2), (8, 2),
                (6, 4), (6, 5),
                (7, 8), (6, 8), (5, 8), (4, 8), (3, 8),
                (8, 7), (2, 7), (7,9)
            ],
            "objective": (8, 8),
            "start_pos": (0, 0)
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
    # Distância inicial
    initial_dist = abs(start_x - obj_x) + abs(start_y - obj_y)
    min_dist = initial_dist

    reached_objective = False
    steps_taken = 0

    for _ in range(SIMULATION_STEPS):
        win = sim.executa()
        steps_taken += 1

        dist = abs(agent.x - obj_x) + abs(agent.y - obj_y)
        if dist < min_dist: min_dist = dist

        if win:
            reached_objective = True
            min_dist = 0  # Distância zero se ganhou
            break

    # Fitness Function
    # Recompensa baseada na aproximação (Max 100 pontos por aproximação)
    improvement = initial_dist - min_dist
    performance_score = improvement * 5

    if reached_objective:
        performance_score += 100  # Bónus gigante por ganhar
        performance_score += (SIMULATION_STEPS - steps_taken) * 2  # Bónus de velocidade
    else:
        # Penalidade por não mexer
        if len(agent.path) < 2:
            performance_score -= 50

    return {
        "performance": performance_score,
        "final_pos": (agent.x, agent.y),
        "path": agent.path,
        "reached": reached_objective
    }

def create_random_weights():
    # Agora com 11 inputs em vez de 9
    return {
        'up': [random.uniform(-1, 1) for _ in range(11)],
        'down': [random.uniform(-1, 1) for _ in range(11)],
        'left': [random.uniform(-1, 1) for _ in range(11)],
        'right': [random.uniform(-1, 1) for _ in range(11)]
    }

def mutate(weights):
    new_weights = copy.deepcopy(weights)
    for action in new_weights:
        for i in range(len(new_weights[action])):
            if random.random() < MUTATION_RATE:
                new_weights[action][i] += random.gauss(0, MUTATION_STRENGTH)
                # Clamp weights
                new_weights[action][i] = max(-5, min(5, new_weights[action][i]))
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
    novelty_scores = []
    behavior_vectors = [res['behavior_vector'] for res in population_results]

    for i, my_behavior in enumerate(behavior_vectors):
        distances = []
        for j, other_behavior in enumerate(behavior_vectors):
            if i == j: continue

            # Distância Euclidiana entre os vetores de comportamento (posições finais em cada cenário)
            dist = 0
            for k in range(len(my_behavior)):
                p1 = my_behavior[k]
                p2 = other_behavior[k]
                dist += (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            dist = math.sqrt(dist)
            distances.append(dist)

        distances.sort()
        k_nn = distances[:NOVELTY_K]
        sparsity = sum(k_nn) / len(k_nn) if k_nn else 0
        novelty_scores.append(sparsity)

    return novelty_scores

def main():
    print(f"--- INICIANDO TREINO (Novelty Search) ---")
    print(f"Pop: {POPULATION_SIZE} | Gens: {GENERATIONS} | Inputs Neurais: 11")

    population = [create_random_weights() for _ in range(POPULATION_SIZE)]
    best_overall_weights = None
    best_overall_score = -float('inf')

    for gen in range(GENERATIONS):
        pop_results = []

        # 1. Avaliação
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

        # 2. Novelty
        novelty_scores = calculate_novelty(pop_results)

        # 3. Fitness (Performance + Novelty)
        final_scored_pop = []
        gen_best_perf = -float('inf')

        for i, res in enumerate(pop_results):
            # Normalizar Novelty pode ajudar, mas aqui somamos direto com peso
            fitness = res["performance"] + (novelty_scores[i] * NOVELTY_WEIGHT)
            final_scored_pop.append((fitness, res["weights"], res["performance"]))

            if res["performance"] > gen_best_perf:
                gen_best_perf = res["performance"]

        # Ordenar por Fitness
        final_scored_pop.sort(key=lambda x: x[0], reverse=True)

        # Guardar melhor absoluto
        if gen_best_perf > best_overall_score:
            best_overall_score = gen_best_perf
            # Encontrar o conjunto de pesos com melhor performance pura nesta geração
            best_entry = max(final_scored_pop, key=lambda x: x[2])
            best_overall_weights = best_entry[1]

        print(f"Gen {gen + 1}: Best Fitness={final_scored_pop[0][0]:.1f} | Best Perf={gen_best_perf:.1f}")

        # 4. Seleção e Reprodução
        survivors = [x[1] for x in final_scored_pop[:int(POPULATION_SIZE * 0.2)]]  # Top 20%
        new_population = copy.deepcopy(survivors)  # Elitismo

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