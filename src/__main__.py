from imports import *
from functions import *
from data_loader import *
from RewardFunctions import *
from CustomCallback import *
from HyperParameter_Tuning import *
from Train_and_Evaluation import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


BUILDING_COUNT=7
DAY_COUNT=7
RANDOM_SEED=0
WEEK = 167

BUILDINGS = [13,4,3,16,2,17,9]
TRAINSTART = 7561
#-----------------------------------

TRAIN_B = BUILDINGS
TEST_B=[17, 13, 4, 10, 16, 1, 6]
TRAINEND= TRAINSTART + WEEK

ACTIVE_OBSERVATIONS =  ['hour','day_type', 'month',
                       'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h',
                       'solar_generation', 'net_electricity_consumption','non_shiftable_load','electrical_storage_soc','carbon_intensity',    
                       'electricity_pricing', 'electricity_pricing_predicted_6h', 'electricity_pricing_predicted_12h' 
                       ]
OFFSET=0
# ---------------------------------------------------------------Trainging Data---------------------------------------------------
train_schema, train_buildings, train_simulation_start_time_step, train_simulation_end_time_step, train_active_observations, train_ROOT_DIRECTORY, train_RANDOM_SEED, train_BUILDING_COUNT, train_DAY_COUNT =\
      load_citylearn_data(random_seed=RANDOM_SEED, building_count=BUILDING_COUNT, day_count=DAY_COUNT,active_observations=ACTIVE_OBSERVATIONS, 
                          forced_selected_buildings=TRAIN_B,forced_simulation_start_time_step=TRAINSTART, forced_simulation_end_time_step = TRAINEND, Prints=False)

# ---------------------------------------------------------------Evaluation Data----------------------------------------------
#Δημιουργία schema-τος για evaluation σε test data
eval_schema,*_ = load_citylearn_data(random_seed=RANDOM_SEED, building_count=BUILDING_COUNT, day_count=DAY_COUNT,active_observations=ACTIVE_OBSERVATIONS, 
                          forced_selected_buildings=TEST_B,forced_simulation_start_time_step=1, forced_simulation_end_time_step = 8759, Prints=False)

# -------------------------------------------------------------Enviroments --------------------------------------
train_env = CityLearnEnv(train_schema, central_agent= True)
eval_env = CityLearnEnv(eval_schema, central_agent= True)

#----------------------------------------------------------------------HyperParameter Tunning--------------------------------------
#best_agent_kwargs=HyperParameter_Tuning(episodes=10,ES_max_no_improve=30 ,pruning_freq=10,episodes_to_check=5, trainEnv= train_env, evalEnv=eval_env,Random_Seed=train_RANDOM_SEED)
#--------------------------------------------------------------------------------SAC--------------------------------------------------------
default_agent_kwargs = {'learning_rate': 0.0003,'buffer_size': 1000000,'learning_starts': 100,'batch_size': 256,'tau': 0.005,'gamma': 0.99,'train_freq': 1}
hp_50_agent_kwargs=  {'learning_rate': 9.465246908441301e-05, 'buffer_size': 50000, 'batch_size': 256, 'tau': 0.008828229777950888, 'gamma': 0.9817169120412357}      
hp_300_agent_kwargs = {'learning_rate': 0.00009330026692728367, 'buffer_size': 50000, 'batch_size': 256, 'tau': 0.005742814302052277, 'gamma': 0.9779423928978465}    
hp_1200_agent_kwargs ={'learning_rate': 0.000138963526275258, 'buffer_size': 50000, 'batch_size': 256, 'tau': 0.005894248557670813, 'gamma': 0.9061070507760203}

save_path = 'model/'
save_name = None
load_path = None 
episodes = 50
reward_function = CustomRewardFunction

start_time = datetime.now(timezone.utc)
# Train Agent
model, _ , _= train_agent(
    env= train_env, 
    agent_kwargs=hp_50_agent_kwargs,
    episodes=episodes,
    reward_function=reward_function,
    random_seed=train_RANDOM_SEED,
    save_path=save_path,
    save_name=save_name,
    load_path=load_path
    
)
#-------------------------------------------------------------------- Evaluation--------------------------------------------------------
eval_rewards, kpis, actions_list, rewards_list,average_kpi = evaluate_agent(
    model, eval_env, reward_function, show_figures=False 
)
print(f"Evaluation KPIs: {average_kpi}")

elapsed_time = datetime.now(timezone.utc) - start_time
print(f"Elapsed Time: {elapsed_time}")

