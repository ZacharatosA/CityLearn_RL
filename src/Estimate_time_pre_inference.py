import time
from imports import*
from data_loader import load_citylearn_data
 
ACTIVE_OBSERVATIONS =  ['hour','day_type', 'month',
                       'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h',  
                       'solar_generation', 'net_electricity_consumption','non_shiftable_load','electrical_storage_soc','carbon_intensity',    
                       'electricity_pricing', 'electricity_pricing_predicted_6h', 'electricity_pricing_predicted_12h',
                       ]
model_path = 'models/50ep/FinalH/sac_citylearn_final.zip'
eval_schema,*_ = load_citylearn_data(random_seed=0, building_count=7, day_count=7,active_observations=ACTIVE_OBSERVATIONS, 
                          forced_selected_buildings=[17, 13, 4, 10, 16, 1, 6],forced_simulation_start_time_step=0, forced_simulation_end_time_step =1,Prints=False)

#---------------------------------------------------------------------
env = CityLearnEnv(schema=eval_schema, central_agent=True)
env = NormalizedObservationWrapper(env)
env = StableBaselines3Wrapper(env)

model = SAC.load(model_path, env=env)
observations = env.reset()

start_time = time.time()
actions, _ = model.predict(observations, deterministic=True)
end_time = time.time()

inference_time = end_time - start_time
print(f"Actions: {actions}")
print(f"Time per inference: {inference_time:.6f} seconds")
