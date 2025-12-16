import random
import json
import copy
import math
from pathlib import Path

from main import Ambiente, Obstacle, Objective, NeuralAgent, CircularSensor, Simulador, Farol, FarolSensor

# Detect agent input size dynamically so training scripts stay in sync with main.py
_TEMP_AGENT = NeuralAgent(0, 0, "N")
INPUT_SIZE = _TEMP_AGENT.input_size
# Added plotting dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    plt = None
    np = None

POPULATION_SIZE = 100
GENERATIONS = 100  # Reduzi um pouco para veres resultados mais depressa
SIMULATION_STEPS = 20
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
ENV_SIZE = 10

# Checkpoint / plotting directories
CHECKPOINT_DIR = Path("checkpoints")
PLOTS_DIR = Path("plots")
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Novelty Parameters
NOVELTY_K = 15
NOVELTY_WEIGHT = 0.5

ENV_SCENARIOS = [
    {"obstacles": [(2, 2), (3, 2), (4, 2)], "objective": (5, 5), "start_pos": (0, 0), "farol": False },
    {"obstacles": [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)], "objective": (8, 2), "start_pos": (2, 2), "farol": False},
    {"obstacles": [(3, 3), (4, 4), (5, 5), (6, 6)], "objective": (9, 9), "start_pos": (0, 0), "farol": False},
    {"obstacles": [(3, 3), (4, 3), (5, 3), (3, 4), (5, 4), (3, 5), (4, 5), (5, 5)], "objective": (4, 4),
     "start_pos": (0, 0), "farol": False},
    {"obstacles": [], "objective": (7, 7), "start_pos": (1, 1), "farol": False},
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
        "farol": True
    },
    {
            "obstacles": [
                (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),(8,1), (9,1),
                (1, 2),
                (1, 3), (3, 3), (4, 3), (5, 3), (6, 3), (8, 3),
                (1, 4), (6, 4), (8, 4),
                (1, 5), (3, 5), (4, 5), (5, 5), (6, 5), (8, 5),
                (1, 6), (8, 6),
                (1, 7), (3, 7), (4, 7), (5, 7), (6, 7), (8, 7),
                (3, 8), (8, 8),
                (1, 9), (8, 9),
            ],
            "objective": (9,9),
            "start_pos": (0,0),
            "farol": False
        },

    {
            "obstacles": [
                (1, 0),
                (1,1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1), (9, 1),
                (1, 2),
                (1, 3), (3, 3), (4, 3), (5, 3), (6, 3), (8, 3),
                (1, 4), (6, 4), (8, 4),
                (1, 5), (3, 5), (4, 5), (5, 5), (6, 5), (8, 5),
                (1, 6), (8, 6),
                (1, 7), (3, 7), (4, 7), (5, 7), (6, 7), (8, 7),
                (3, 8), (8, 8),
                (1, 9), (8, 9),
            ],
            "objective": (9, 9),
            "start_pos": (0, 0),
            "farol": False
    }
]


def create_environment(scenario, agent: NeuralAgent | None = None):
    """Create an Ambiente for the given scenario.

    If `agent` is provided, the function will install the appropriate
    sensors on that agent (so the caller's agent is correctly instrumented
    according to whether the scenario has a farol). The Ambiente is
    returned; callers are expected to add their agent (Simulador does this).
    """
    amb = Ambiente(10)
    for obs_pos in scenario["obstacles"]:
        amb.add_object(Obstacle(obs_pos[0], obs_pos[1]))

    isfarol = scenario["farol"]
    obj_pos = scenario["objective"]

    # Add the target object to the environment
    if isfarol:
        farol = Farol(obj_pos[0], obj_pos[1])
        amb.add_object(farol)
        if agent is not None:
            agent.instala(CircularSensor(1))
            agent.instala(FarolSensor(farol))
    else:
        amb.add_object(Objective(obj_pos[0], obj_pos[1]))
        if agent is not None:
            agent.instala(CircularSensor(3))

    return amb


def run_episode(weights, scenario):
    start_x, start_y = scenario["start_pos"]
    agent = NeuralAgent(start_x, start_y, "N")
    agent.set_weights(weights)

    # Create environment and install the correct sensors on our agent
    amb = create_environment(scenario, agent)
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
        "steps": steps_taken,
        "reached": reached_objective
    }


def create_random_weights():
    # Create weight vectors sized to the agent's current input dimension
    return {
        'up': [random.uniform(-1, 1) for _ in range(INPUT_SIZE)],
        'down': [random.uniform(-1, 1) for _ in range(INPUT_SIZE)],
        'left': [random.uniform(-1, 1) for _ in range(INPUT_SIZE)],
        'right': [random.uniform(-1, 1) for _ in range(INPUT_SIZE)]
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
    print(f"Pop: {POPULATION_SIZE} | Gens: {GENERATIONS} | Inputs Neurais: {INPUT_SIZE}")

    population = [create_random_weights() for _ in range(POPULATION_SIZE)]
    best_overall_weights = None
    best_overall_score = -float('inf')

    # History collectors for plotting
    best_fitness_history = []
    avg_fitness_history = []
    best_perf_history = []
    avg_perf_history = []
    novelty_avg_history = []

    actions = ['up', 'down', 'left', 'right']
    # mean weights history: action -> input_index -> list over generations
    mean_weights_history = {a: [[] for _ in range(INPUT_SIZE)] for a in actions}

    # Record the best individual's per-scenario paths for each generation
    best_paths_history = []

    for gen in range(GENERATIONS):
        pop_results = []

        # 1. Avaliação
        for weights in population:
            # Create a single learning agent per genome so online RL updates
            # accumulate into the final weights evaluated by the GA.
            total_perf = 0
            behavior_vector = []
            paths = []

            # Start agent with genome weights
            # Position will be reset per scenario below
            # Use a deep copy of weights so we don't mutate the original population entry
            agent = NeuralAgent(0, 0, "N")
            agent.set_weights(copy.deepcopy(weights))
            # Enable some exploration/learning during evaluation
            agent.epsilon = 0.2
            agent.learning_rate = 0.1

            for scenario in ENV_SCENARIOS:
                # Reset agent state for this scenario (position/path/sensors) but keep weights
                start_x, start_y = scenario["start_pos"]
                agent.x = start_x
                agent.y = start_y
                agent.path = []
                agent.visit_counts = {(start_x, start_y): 1}
                agent.recent_positions = []
                agent.sensor = []

                amb = create_environment(scenario, agent)
                sim = Simulador([agent], amb)

                obj_x, obj_y = scenario["objective"]
                initial_dist = abs(start_x - obj_x) + abs(start_y - obj_y)
                min_dist = initial_dist
                reached_objective = False

                for _ in range(SIMULATION_STEPS):
                    win = sim.executa()
                    dist = abs(agent.x - obj_x) + abs(agent.y - obj_y)
                    if dist < min_dist:
                        min_dist = dist
                    if win:
                        reached_objective = True
                        min_dist = 0
                        break

                improvement = initial_dist - min_dist
                performance_score = improvement * 5
                if reached_objective:
                    performance_score += 100

                total_perf += performance_score
                behavior_vector.append((agent.x, agent.y))
                # Save the visited path (list of (x,y)) for this scenario
                paths.append(copy.deepcopy(agent.path))

            avg_perf = total_perf / len(ENV_SCENARIOS)
            # Use the agent's post-learning weights as the representative genome
            final_weights = copy.deepcopy(agent.weights)
            pop_results.append({
                "weights": final_weights,
                "performance": avg_perf,
                "behavior_vector": behavior_vector,
                "paths": paths
            })

        # 2. Novelty
        novelty_scores = calculate_novelty(pop_results)

        # 3. Fitness (Performance + Novelty)
        final_scored_pop = []
        gen_best_perf = -float('inf')

        for i, res in enumerate(pop_results):
            # Normalizar Novelty pode ajudar, mas aqui somamos direto com peso
            fitness = res["performance"] + (novelty_scores[i] * NOVELTY_WEIGHT)
            # Include the whole res so we can access paths later
            final_scored_pop.append((fitness, res["weights"], res["performance"], res))

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

        # Record the top individual's paths for this generation (aggregate of all scenarios)
        if final_scored_pop:
            top_res = final_scored_pop[0][3]
            best_paths_history.append(top_res.get('paths', []))
        else:
            best_paths_history.append([])

        # --- Metrics for checkpointing & plotting ---
        fitness_values = [x[0] for x in final_scored_pop]
        perf_values = [x[2] for x in final_scored_pop]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        avg_perf = sum(perf_values) / len(perf_values)

        best_fitness_history.append(fitness_values[0])
        avg_fitness_history.append(avg_fitness)
        best_perf_history.append(gen_best_perf)
        avg_perf_history.append(avg_perf)
        novelty_avg_history.append(sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0)

        # Compute mean weights across current population for each action/input
        for action in actions:
            for idx in range(INPUT_SIZE):
                vals = [ind[action][idx] for ind in population]
                mean_weights_history[action][idx].append(sum(vals) / len(vals))

        print(f"Gen {gen + 1}: Best Fitness={final_scored_pop[0][0]:.1f} | Best Perf={gen_best_perf:.1f}")

        # Save checkpoint (top 5 weights of the generation)
        checkpoint = {
            'generation': gen + 1,
            'top_weights': [entry[1] for entry in final_scored_pop[:5]],
            'top_fitness': final_scored_pop[0][0],
            'avg_fitness': avg_fitness,
            'best_perf': gen_best_perf,
        }
        with open(CHECKPOINT_DIR / f"checkpoint_gen_{gen+1}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

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

    # Save best overall weights
    if best_overall_weights:
        with open("best_weights.json", "w") as f:
            json.dump(best_overall_weights, f)
        print("Saved best weights to best_weights.json")

    # --- Plotting ---
    if plt is None:
        print("matplotlib not available; skipping plots. Install matplotlib to enable plots.")
        return

    gens = list(range(1, GENERATIONS + 1))

    # Fitness/Performance plot
    plt.figure()
    plt.plot(gens, best_fitness_history, label='Best Fitness')
    plt.plot(gens, avg_fitness_history, label='Avg Fitness')
    plt.plot(gens, best_perf_history, label='Best Perf')
    plt.plot(gens, avg_perf_history, label='Avg Perf')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Fitness and Performance over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'fitness_performance.png')
    plt.close()

    # Novelty plot
    plt.figure()
    plt.plot(gens, novelty_avg_history, label='Avg Novelty')
    plt.xlabel('Generation')
    plt.ylabel('Novelty (avg)')
    plt.title('Average Novelty over Generations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'novelty.png')
    plt.close()

    # Weights evolution plots (mean per-action per-input)
    for action in actions:
        plt.figure()
        for idx in range(INPUT_SIZE):
            plt.plot(gens, mean_weights_history[action][idx], label=f'input_{idx}')
        plt.xlabel('Generation')
        plt.ylabel('Mean Weight')
        plt.title(f'Mean Weights for action: {action}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'weights_mean_{action}.png')
        plt.close()

    # Heatmap(s): cumulative over generations showing aggregated visit counts of best individuals
    # Each saved image for generation N aggregates paths from best individuals of generations 1..N
    if np is not None:
        # Build a single cumulative grid that aggregates visits from the best individual of each generation
        cumulative_grid = np.zeros((ENV_SIZE, ENV_SIZE), dtype=float)
        for gen_paths in best_paths_history:
            for scenario_path in gen_paths:
                if not scenario_path:
                    continue
                for pos in scenario_path:
                    try:
                        x, y = pos
                    except Exception:
                        continue
                    if 0 <= x < ENV_SIZE and 0 <= y < ENV_SIZE:
                        cumulative_grid[y, x] += 1.0

        # Save only the final combined heatmap (no per-generation snapshots)
        plt.figure(figsize=(6, 6))
        plt.imshow(cumulative_grid, cmap='hot', origin='lower')
        plt.colorbar(label='Cumulative Visit Count')
        plt.title('Final cumulative heatmap (all generations)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'heatmap_cumulative_all_generations.png')
        plt.close()
    else:
         print('numpy not available; skipping heatmap generation.')

    # Save a small summary JSON (metrics) for external analysis
    summary = {
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'best_perf_history': best_perf_history,
        'avg_perf_history': avg_perf_history,
        'novelty_avg_history': novelty_avg_history,
        'generations': GENERATIONS
    }
    with open(CHECKPOINT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plots to {PLOTS_DIR} and checkpoints to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()