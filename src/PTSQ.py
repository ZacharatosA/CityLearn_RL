import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, quantize_dynamic
from data_loader import*


# Ο κώδικας σας για τη δημιουργία του περιβάλλοντος
BUILDINGS = [4, 2, 17, 1, 13, 11, 10]
TRAIN_B = BUILDINGS
ACTIVE_OBSERVATIONS =  ['hour','day_type', 'month',#'daylight_savings_status',
                       'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h',
                       'solar_generation', 'net_electricity_consumption','non_shiftable_load','electrical_storage_soc','carbon_intensity',    
                       'electricity_pricing', 'electricity_pricing_predicted_6h', 'electricity_pricing_predicted_12h' 
                       ]
OFFSET=0

# Φόρτωση δεδομένων για την εκπαίδευση
schema, _, _, _, _, _, _, _, _ =\
      load_citylearn_data(random_seed=0, building_count=7, day_count=7,active_observations=ACTIVE_OBSERVATIONS, 
                          forced_selected_buildings=TRAIN_B,forced_simulation_start_time_step=0, forced_simulation_end_time_step=8759)
env = CityLearnEnv(schema=schema, central_agent=True)
env = NormalizedObservationWrapper(env)
env = StableBaselines3Wrapper(env)

# Συλλογή των calibration_data
calibration_data = []

obs = env.reset()
calibration_data.append(np.array(obs, dtype=np.float32).reshape(1, -1))

for _ in range(8759):
    action = env.action_space.sample()
    obs, _, done, _ = env.step(action)
    calibration_data.append(np.array(obs, dtype=np.float32).reshape(1, -1))
    if done:
        obs = env.reset()

# Αποθήκευση των calibration_data σε αρχείο .npy
np.save('calibration_data.npy', calibration_data)
class DataReaderForCalibration(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = calibration_data
        self.iterator = iter(self.data)

    def get_next(self):
        try:
            input = next(self.iterator)
            return {'input': input}
        except StopIteration:
            return None


# Δημιουργία του DataReader
data_reader = DataReaderForCalibration(calibration_data)

# Εκτέλεση της κβαντοποίησης
model_fp32 = 'model_nD.onnx'
model_quant = 'model_nD_quant.onnx'

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
)