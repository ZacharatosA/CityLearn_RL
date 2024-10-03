from imports import *
from functions import *
from data_loader import load_citylearn_data
from CustomCallback import*

def train_agent(
    env: Mapping[str, CityLearnEnv],
    agent_kwargs: dict,
    episodes: int,
    reward_function: RewardFunction,
    random_seed: int,
    save_path: str = None,
    save_name: str = None,
    load_path: str = None,
    reference_envs: Mapping[str, CityLearnEnv] = None,
    trial=None, 
    pruning_freq: int = 100,
    episodes_to_check: int=100
) -> tuple:

    env.reward_function = reward_function(env=env)
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    if load_path is not None:
        model = SAC.load(load_path, env=env)
    else:
        model = SAC('MlpPolicy', env, **agent_kwargs, seed=random_seed)

    total_timesteps = episodes * (env.time_steps - 1)
    #print('Number of episodes to train:', episodes)
    #print('Chosen Hyperparameters:', agent_kwargs)

    with tqdm(total=total_timesteps, desc="Training Progress" ) as pbar:
        callback = CustomCallback(env=env, loader=pbar, trial=trial, pruning_freq=pruning_freq,episodes_to_check=episodes_to_check)  
        checkpoint_callback = None
        if save_path and save_name:
            print(f"Model Saved at {save_path}/{save_name}_final.zip")
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_path, name_prefix=save_name)
            callbacks = [callback, checkpoint_callback]
        else:
            callbacks = [callback]

        train_start_timestamp = datetime.now(timezone.utc)
        try:
            model = model.learn(total_timesteps=total_timesteps, callback=callbacks)
        except optuna.exceptions.TrialPruned:
            return None, env, None
        train_end_timestamp = datetime.now(timezone.utc)

    if save_path and save_name:
        model.save(f"{save_path}/{save_name}_final")

    reward_history = callback.reward_history

    return model, env, reward_history

def evaluate_agent(model, env, reward_function, show_figures: bool = None):
    env.reward_function = reward_function(env=env)
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    observations = env.reset()
    actions_list = []
    rewards_list = []

    while not env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, rewards, _, _ = env.step(actions)
        actions_list.append(actions.tolist())
        rewards_list.append(rewards)

    eval_rewards = np.sum(rewards_list)
    kpis = env.evaluate()
    
    # Remove'zero_net_energy'
    kpis = kpis[kpis['cost_function'] != 'zero_net_energy']
    #Average District KPIs
    district_kpis = kpis[kpis['level'] == 'district']
    average_district_kpi_value = district_kpis['value'].mean()

    # Import Average District KPIs 
    average_kpi = pd.DataFrame({
        'cost_function': ['Average_District_KPIs'],
        'value': [average_district_kpi_value],
        'name': ['Average_District_KPIs'],
        'level': ['district']
    })
    # Update DataFrame των KPIs
    kpis = pd.concat([kpis, average_kpi], ignore_index=True)

    if show_figures:
        plot_simulation_summary({'Evaluation': env})

    return eval_rewards, kpis, actions_list, rewards_list,round(average_district_kpi_value ,7)
