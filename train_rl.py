import random
import json
import copy
from pathlib import Path

from main import Ambiente, Obstacle, Objective, NeuralAgent, CircularSensor, Simulador, Farol, FarolSensor

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    plt = None
    np = None

# Training hyperparameters
EPISODES = 10000  # increased default number of episodes for RL training
# Increase steps per episode to give agent more time to reach/see the objective
SIMULATION_STEPS = 50
# Stronger exploration and slower decay so agent explores more during early training
EPSILON_START = 0.9
EPSILON_END = 0.02
EPSILON_DECAY = 0.9995
CHECKPOINT_INTERVAL = 200  # episodes between checkpoints
TOP_K = 5  # keep top-K weight snapshots by episode performance

# Learning rate schedule (start larger, decay)
LR_START = 0.3
LR_END = 0.05
LR_DECAY = (LR_END / LR_START) ** (1.0 / max(1, EPISODES))

# Directories
CHECKPOINT_DIR = Path("rl_checkpoints")
PLOTS_DIR = Path("rl_plots")
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Scenarios (reuse some of the sets used elsewhere)
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
        "farol": Farol
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
    """Build Ambiente for a scenario. If `agent` is supplied, install appropriate sensors (and clear old ones)."""
    amb = Ambiente(10)
    for obs_pos in scenario["obstacles"]:
        amb.add_object(Obstacle(obs_pos[0], obs_pos[1]))

    isfarol = scenario["farol"]
    obj_pos = scenario["objective"]

    if isfarol:
        farol = Farol(obj_pos[0], obj_pos[1])
        amb.add_object(farol)
        if agent is not None:
            agent.sensor = []
            agent.instala(CircularSensor(1))
            agent.instala(FarolSensor(farol))
    else:
        amb.add_object(Objective(obj_pos[0], obj_pos[1]))
        if agent is not None:
            agent.sensor = []
            agent.instala(CircularSensor(3))

    return amb


def reset_agent_for_episode(agent: NeuralAgent, start_pos: tuple):
    agent.x, agent.y = start_pos
    agent.path = []
    agent.visit_counts = {(agent.x, agent.y): 1}
    agent.recent_positions = []
    agent.last_state_inputs = None
    agent.last_direction = None


def evaluate_episode(agent: NeuralAgent, scenario, render=False):
    start_x, start_y = scenario["start_pos"]
    reset_agent_for_episode(agent, (start_x, start_y))

    amb = create_environment(scenario, agent)
    sim = Simulador([agent], amb)

    obj_x, obj_y = scenario["objective"]
    initial_dist = abs(start_x - obj_x) + abs(start_y - obj_y)
    min_dist = initial_dist
    reached = False
    steps_taken = 0

    for _ in range(SIMULATION_STEPS):
        win = sim.executa()
        steps_taken += 1
        dist = abs(agent.x - obj_x) + abs(agent.y - obj_y)
        if dist < min_dist:
            min_dist = dist
        if win:
            reached = True
            min_dist = 0
            break

    improvement = initial_dist - min_dist
    performance_score = improvement * 5
    if reached:
        performance_score += 100
        performance_score += (SIMULATION_STEPS - steps_taken) * 2
    else:
        if len(agent.path) < 2:
            performance_score -= 50

    return {
        "performance": performance_score,
        "reached": reached,
        "steps": steps_taken,
        "final_pos": (agent.x, agent.y),
        "path": list(agent.path)
    }


def save_checkpoint(agent: NeuralAgent, episode: int, metrics: dict, top_k_list=None):
    payload = {
        'episode': episode,
        'weights': agent.weights,
        'metrics': metrics,
    }
    if top_k_list is not None:
        # store summarized top-k metadata (episode, performance)
        payload['top_k'] = [{'episode': t['episode'], 'performance': t['performance']} for t in top_k_list]
    with open(CHECKPOINT_DIR / f"checkpoint_ep_{episode}.json", "w") as f:
        json.dump(payload, f, indent=2)


def main():
    print("--- RL-ONLY TRAINING START ---")

    agent = NeuralAgent(0, 0, "N")

    # Try to load existing best weights if present
    if Path("best_weights.json").exists():
        try:
            with open("best_weights.json", "r") as f:
                w = json.load(f)
            agent.set_weights(w)
            print("Loaded existing weights into RL agent")
        except Exception as e:
            print(f"Could not load weights: {e}")

    # Tracking
    perf_history = []
    success_history = []
    steps_history = []
    eps_history = []

    # For weights tracking (mean per action/input)
    actions = agent.actions
    input_size = agent.input_size
    mean_weights_history = {a: [[] for _ in range(input_size)] for a in actions}

    # Top-K list for best-performing episodes
    top_k_list = []  # each entry: {'episode': int, 'performance': float, 'weights': {...}}

    epsilon = EPSILON_START
    agent.epsilon = epsilon
    # Set agent RL hyperparameters explicitly
    agent.learning_rate = LR_START
    agent.discount_factor = 0.95

    best_performance = float('-inf')

    for ep in range(1, EPISODES + 1):
        # Anneal epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        agent.epsilon = epsilon
        # Decay learning rate (schedule)
        agent.learning_rate = max(LR_END, agent.learning_rate * LR_DECAY)

        # Choose a random scenario for this episode
        scenario = random.choice(ENV_SCENARIOS)

        result = evaluate_episode(agent, scenario)

        perf = result['performance']

        if perf > best_performance:
            best_performance = perf

        perf_history.append(perf)
        success_history.append(1 if result['reached'] else 0)
        steps_history.append(result['steps'])
        eps_history.append(epsilon)

        # record mean weights
        for a in actions:
            for i in range(input_size):
                mean_weights_history[a][i].append(agent.weights[a][i])

        # Update top-K snapshots if this episode is among the best
        inserted = False
        if len(top_k_list) < TOP_K:
            top_k_list.append({'episode': ep, 'performance': perf, 'weights': copy.deepcopy(agent.weights)})
            inserted = True
        else:
            # find current worst in top_k
            worst_perf = min(top_k_list, key=lambda x: x['performance'])['performance']
            if perf > worst_perf:
                # replace the worst one
                # remove first worst occurrence
                for idx, entry in enumerate(top_k_list):
                    if entry['performance'] == worst_perf:
                        top_k_list.pop(idx)
                        break
                top_k_list.append({'episode': ep, 'performance': perf, 'weights': copy.deepcopy(agent.weights)})
                inserted = True

        if inserted:
            # sort descending by performance
            top_k_list.sort(key=lambda x: x['performance'], reverse=True)
            # persist top-k summary to disk
            with open(CHECKPOINT_DIR / 'top_k_weights.json', 'w') as f:
                json.dump([{'episode': t['episode'], 'performance': t['performance'], 'weights': t['weights']} for t in top_k_list], f, indent=2)

        # periodic checkpointing including top-k summary
        if ep % CHECKPOINT_INTERVAL == 0 or ep == 1 or ep == EPISODES:
            metrics = {
                'episode': ep,
                'mean_perf': sum(perf_history[-CHECKPOINT_INTERVAL:]) / min(len(perf_history), CHECKPOINT_INTERVAL),
                'success_rate': sum(success_history[-CHECKPOINT_INTERVAL:]) / min(len(success_history), CHECKPOINT_INTERVAL),
                'epsilon': epsilon
            }
            save_checkpoint(agent, ep, metrics, top_k_list=top_k_list)
            print(f"Saved checkpoint at episode {ep}: mean_perf={metrics['mean_perf']:.2f} success_rate={metrics['success_rate']:.2f}")

    # Save final weights (prefer the top-1 if available)
    if top_k_list:
        best_weights = top_k_list[0]['weights']
    else:
        best_weights = agent.weights

    with open("best_weights_rl.json", "w") as f:
        json.dump(best_weights, f)
    print("Saved final RL weights to best_weights_rl.json")

    # Also ensure top_k_weights.json exists (may have been created earlier)
    if not (CHECKPOINT_DIR / 'top_k_weights.json').exists():
        with open(CHECKPOINT_DIR / 'top_k_weights.json', 'w') as f:
            json.dump([{'episode': t['episode'], 'performance': t['performance'], 'weights': t['weights']} for t in top_k_list], f, indent=2)

    # Save training summary
    summary = {
        'perf_history': perf_history,
        'success_history': success_history,
        'steps_history': steps_history,
        'eps_history': eps_history,
        'episodes': EPISODES
    }
    with open(CHECKPOINT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plotting
    if plt is None:
        print("matplotlib not available; skipping plots. Install matplotlib to enable plots.")
        return

    episodes = list(range(1, EPISODES + 1))

    plt.figure()
    plt.plot(episodes, perf_history, label='Performance per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Performance')
    plt.title('RL Performance over Episodes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'performance.png')
    plt.close()

    # Success rate moving average
    window = max(1, EPISODES // 20)
    success_arr = np.array(success_history)
    ma = np.convolve(success_arr, np.ones(window) / window, mode='valid')
    plt.figure()
    plt.plot(list(range(window, EPISODES + 1)), ma, label=f'Success rate (MA window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Moving Average')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'success_ma.png')
    plt.close()

    # Epsilon
    plt.figure()
    plt.plot(episodes, eps_history, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'epsilon.png')
    plt.close()

    # Mean weights evolution
    for a in actions:
        plt.figure()
        for i in range(input_size):
            plt.plot(episodes, mean_weights_history[a][i], label=f'input_{i}')
        plt.xlabel('Episode')
        plt.ylabel('Weight')
        plt.title(f'Mean weights evolution for action: {a}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'weights_{a}.png')
        plt.close()

    # --- New: evaluation of trained agent vs random agent over multiple runs ---
    try:
        N_EVAL_RUNS = 100
        trained_perfs = []
        trained_success = []
        trained_steps = []
        random_perfs = []
        random_success = []
        random_steps = []

        # Helper to create random weights matching input size
        def create_random_weights(sz):
            return {act: [random.uniform(-1, 1) for _ in range(sz)] for act in actions}

        # Use the best weights from earlier (prefer top-K best if exists)
        if top_k_list:
            eval_best_weights = top_k_list[0]['weights']
        else:
            eval_best_weights = best_weights

        for i in range(N_EVAL_RUNS):
            scenario = random.choice(ENV_SCENARIOS)

            # Trained agent (fresh instance to avoid any learning during eval)
            t_agent = NeuralAgent(0, 0, "N")
            t_agent.set_weights(eval_best_weights)
            t_agent.epsilon = 0.0
            t_agent.learning_rate = 0.0
            tres = evaluate_episode(t_agent, scenario)
            trained_perfs.append(tres['performance'])
            trained_success.append(1 if tres['reached'] else 0)
            trained_steps.append(tres['steps'])

            # Random agent
            r_agent = NeuralAgent(0, 0, "N")
            r_agent.set_weights(create_random_weights(input_size))
            r_agent.epsilon = 0.0
            r_agent.learning_rate = 0.0
            rres = evaluate_episode(r_agent, scenario)
            random_perfs.append(rres['performance'])
            random_success.append(1 if rres['reached'] else 0)
            random_steps.append(rres['steps'])

        # Compute summary stats
        if np is not None:
            trained_perf_mean = float(np.mean(trained_perfs))
            random_perf_mean = float(np.mean(random_perfs))
            trained_success_rate = float(np.mean(trained_success))
            random_success_rate = float(np.mean(random_success))
        else:
            trained_perf_mean = sum(trained_perfs) / len(trained_perfs)
            random_perf_mean = sum(random_perfs) / len(random_perfs)
            trained_success_rate = sum(trained_success) / len(trained_success)
            random_success_rate = sum(random_success) / len(random_success)

        # Average steps among successful runs (None if no successes)
        trained_steps_success = [s for s, succ in zip(trained_steps, trained_success) if succ]
        random_steps_success = [s for s, succ in zip(random_steps, random_success) if succ]
        if np is not None:
            trained_avg_steps_success = float(np.mean(trained_steps_success)) if trained_steps_success else None
            random_avg_steps_success = float(np.mean(random_steps_success)) if random_steps_success else None
            # Average over all runs: count failures as SIMULATION_STEPS
            trained_steps_all = [s if succ else SIMULATION_STEPS for s, succ in zip(trained_steps, trained_success)]
            random_steps_all = [s if succ else SIMULATION_STEPS for s, succ in zip(random_steps, random_success)]
            trained_avg_steps_all = float(np.mean(trained_steps_all))
            random_avg_steps_all = float(np.mean(random_steps_all))
        else:
            trained_avg_steps_success = (sum(trained_steps_success) / len(trained_steps_success)) if trained_steps_success else None
            random_avg_steps_success = (sum(random_steps_success) / len(random_steps_success)) if random_steps_success else None
            trained_steps_all = [s if succ else SIMULATION_STEPS for s, succ in zip(trained_steps, trained_success)]
            random_steps_all = [s if succ else SIMULATION_STEPS for s, succ in zip(random_steps, random_success)]
            trained_avg_steps_all = sum(trained_steps_all) / len(trained_steps_all)
            random_avg_steps_all = sum(random_steps_all) / len(random_steps_all)

        # Save raw results
        comparison_payload = {
            'N_runs': N_EVAL_RUNS,
            'trained': {
                'performances': trained_perfs,
                'successes': trained_success,
                'steps': trained_steps,
                'mean_performance': trained_perf_mean,
                'success_rate': trained_success_rate
                , 'avg_steps_to_success': trained_avg_steps_success
                , 'avg_steps_all_runs': trained_avg_steps_all
            },
            'random': {
                'performances': random_perfs,
                'successes': random_success,
                'steps': random_steps,
                'mean_performance': random_perf_mean,
                'success_rate': random_success_rate
                , 'avg_steps_to_success': random_avg_steps_success
                , 'avg_steps_all_runs': random_avg_steps_all
            }
        }
        with open(CHECKPOINT_DIR / 'trained_vs_random_results.json', 'w') as f:
            json.dump(comparison_payload, f, indent=2)

        # Plot comparison (bar chart of mean performance and success rate)
        # Create a 1x3 comparison: mean perf, success rate, avg steps-to-success
        plt.figure(figsize=(12, 4))
        labels = ['Trained', 'Random']

        # Mean performance
        plt.subplot(1, 3, 1)
        perf_vals = [trained_perf_mean, random_perf_mean]
        plt.bar(labels, perf_vals, color=['tab:blue', 'tab:orange'])
        plt.ylabel('Mean Performance')
        plt.title('Mean Performance ({} runs)'.format(N_EVAL_RUNS))

        # Success rate
        plt.subplot(1, 3, 2)
        succ_vals = [trained_success_rate, random_success_rate]
        plt.bar(labels, succ_vals, color=['tab:green', 'tab:red'])
        plt.ylabel('Success Rate')
        plt.title('Success Rate (reach objective)')

        # Avg steps to success (plot only if at least one success exists; show N/A otherwise)
        plt.subplot(1, 3, 3)
        # For plotting, replace None with 0 but annotate with actual value text
        plot_steps = [trained_avg_steps_success if trained_avg_steps_success is not None else 0,
                      random_avg_steps_success if random_avg_steps_success is not None else 0]
        plt.bar(labels, plot_steps, color=['tab:purple', 'tab:gray'])
        plt.ylabel('Avg Steps to Success')
        plt.title('Avg Steps (successful runs only)')
        # Annotate exact values or 'N/A'
        for idx, val in enumerate([trained_avg_steps_success, random_avg_steps_success]):
            text = f"{val:.2f}" if val is not None else "N/A"
            plt.text(idx, (plot_steps[idx] if plot_steps[idx] > 0 else 0.5), text, ha='center', va='bottom')

        # Also log the all-runs averages to console for quick inspection
        print(f"Avg steps (all runs) - Trained: {trained_avg_steps_all:.2f}, Random: {random_avg_steps_all:.2f}")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'trained_vs_random.png')
        plt.close()

    except Exception as e:
        print(f"Error during trained vs random evaluation: {e}")

    print(f"Saved plots to {PLOTS_DIR} and checkpoints to {CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()

