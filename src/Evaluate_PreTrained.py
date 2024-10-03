from imports import *
from functions import *
from data_loader import *
from RewardFunctions import *
from CustomCallback import *
from Train_and_Evaluation import *
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import onnxruntime as ort


BUILDING_COUNT=7
DAY_COUNT=7
RANDOM_SEED=3
WEEK = 167
MONTH =729
YEAR=8759

 
ACTIVE_OBSERVATIONS =  ['hour','day_type', 'month',
                       'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h',  
                       'solar_generation', 'net_electricity_consumption','non_shiftable_load','electrical_storage_soc','carbon_intensity',    
                       'electricity_pricing', 'electricity_pricing_predicted_6h', 'electricity_pricing_predicted_12h',
                       ]
#-----------------------
TRAIN_B =  [13,4,3,16,2,17,9]
TEST_B=  [17, 13, 4, 10, 16, 1, 6]
#TEST_B=  [11, 5, 3, 8, 7, 9, 14]

Value = 1
EVALSTART = 1 #Value*730 - 729
EVALEND =EVALSTART + WEEK   
#EVALEND =EVALSTART + MONTH
EVALEND =YEAR

model_path = 'path_to_your_model'
eval_schema,*_ = load_citylearn_data(random_seed=RANDOM_SEED, building_count=BUILDING_COUNT, day_count=DAY_COUNT,active_observations=ACTIVE_OBSERVATIONS, 
                          forced_selected_buildings=TEST_B,forced_simulation_start_time_step=EVALSTART, forced_simulation_end_time_step =EVALEND,Prints=False)

#---------------------------------------------------------------------

# Create evaluation environment

eval_env = CityLearnEnv(schema=eval_schema, central_agent=True)
eval_env = NormalizedObservationWrapper(eval_env)
eval_env = StableBaselines3Wrapper(eval_env)

model = SAC.load(model_path, env=eval_env)

eval_env = CityLearnEnv(schema=eval_schema, central_agent=True)
reward_function = CustomRewardFunction


eval_rewards, kpis, actions_list, rewards_list,TestKPI = evaluate_agent(model, eval_env, reward_function, show_figures=False  )
print(f"Month: {Value}  ({EVALSTART}-{EVALEND})")
print(f"{TestKPI}")
#print(f"Evaluation Rewards: {eval_rewards}")

