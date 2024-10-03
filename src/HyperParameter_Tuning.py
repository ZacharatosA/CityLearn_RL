from imports import*
from Train_and_Evaluation import *
from CustomCallback import*
from RewardFunctions import *
import logging

def HyperParameter_Tuning(
    episodes: int,
    ES_max_no_improve: int,
    pruning_freq: int,
    episodes_to_check: int,
    trainEnv: Mapping[str, 'CityLearnEnv'],
    evalEnv: Mapping[str, 'CityLearnEnv'],
    Random_Seed: int
):
    # Initial set of hyperparameters
    initial_params = {
        'learning_rate': 0.0003,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
    }

    def objective(trial):
        # Configure hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        buffer_size = trial.suggest_categorical('buffer_size', [50000, 100000, 500000, 1000000])
        batch_size = trial.suggest_categorical('batch_size', [256, 512])
        tau = trial.suggest_float('tau', 0.005, 0.01, log=True)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)

        agent_kwargs = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': 100,  # Fixed value
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': 1,  # Fixed value
        }

        # Simulate training
        print(f"Trial {trial.number} starts:")
        Reward_Function = CustomRewardFunction
        try:
            model, _, reward = train_agent(
                env=trainEnv,
                agent_kwargs=agent_kwargs,
                episodes=episodes,
                reward_function=Reward_Function,
                random_seed=Random_Seed,
                trial=trial,
                pruning_freq=pruning_freq,
                episodes_to_check=episodes_to_check
            )
        except optuna.exceptions.TrialPruned:
            raise optuna.exceptions.TrialPruned()

        if model is None:
            return float('nan')

        # Evaluation
        eval_reward, kpis, reward_list, _ ,average_district_kpi= evaluate_agent(model, evalEnv, Reward_Function, None)
        return -average_district_kpi

    start_time = datetime.now()

    # Configure Optuna logging to show only errors to suppress warnings
    logging.getLogger("optuna").setLevel(logging.ERROR)

    # Custom Callback to print trial results in desired format
    def trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number} finished with value: {trial.value}")
            print(f"Best is trial {study.best_trial.number} with value: {study.best_trial.value}.")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"Trial {trial.number} was pruned.")
            print(f"Best is trial {study.best_trial.number} with value: {study.best_trial.value}.")


    # Initialize Optuna study without loading existing studies and without persistent storage
    study = optuna.create_study(
        direction='maximize',
        load_if_exists=False,  # Do not load existing studies
        pruner=optuna.pruners.HyperbandPruner()
    )

    # Enqueue the initial trial
    study.enqueue_trial(initial_params)

    no_improve_counter = 0      # Counter for trials without improvement
    best_reward = np.NINF       # Initialize best reward
    delta = 0.04                # Improvement threshold

    # Main optimization loop with Early Stopping
    for _ in range(100):
        study.optimize(lambda trial: objective(trial), n_trials=1, callbacks=[trial_callback])
        trial = study.best_trial

        if trial.value > (1 - delta) * best_reward:
            no_improve_counter = 0  # Reset improvement counter
        if trial.value > best_reward:
            if trial.value < (1 - delta) * best_reward:
                no_improve_counter -= 1
            best_reward = trial.value
            print("Best hyperparameters: ", study.best_params)
            print("Early stopping counter:", no_improve_counter)
            print(f"Best Value: {best_reward}, Delta Value:{(1 - delta) * best_reward}")
            print("----------------------------------------")
        else:
            no_improve_counter += 1  # Increment counter for no improvement
            print("Best hyperparameters: ", study.best_params)
            print("Early stopping counter:", no_improve_counter)
            print(f"Best Value: {best_reward}, Delta Value:{(1 - delta) * best_reward}")
            print("----------------------------------------")

        if no_improve_counter >= ES_max_no_improve:
            print("Early stopping triggered.")
            break

    print("Best parameters:", study.best_params)
    elapsed_time = datetime.now() - start_time
    print(f"Total execution time: {elapsed_time}")
    best_agent_kwargs = study.best_params  # Save the best hyperparameters

    return best_agent_kwargs
