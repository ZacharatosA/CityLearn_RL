# data_loader.py
from imports import*
from functions import *

def load_citylearn_data(dataset_name='citylearn_challenge_2022_phase_all' , random_seed= None, building_count=None, day_count= None, active_observations=None,forced_selected_buildings:List[int] = None,forced_simulation_start_time_step : int = None , forced_simulation_end_time_step : int = None, Prints : bool = True):
    
    if random_seed is None:
        random_seed = 0
    if building_count is None:
        building_count=3
    if day_count is None:
        day_count=7
    if active_observations is None:
        active_observations = ['hour', 'day_type', 'month', 'daylight_savings_status' ]
    
    schema = DataSet.get_schema(dataset_name)
    #print('All CityLearn datasets:', sorted(DataSet.get_names()))
    root_directory = schema['root_directory']

    # Διαμόρφωση του schema
    schema, buildings = set_schema_buildings(schema, building_count, random_seed,forced_selected_buildings)
    schema, simulation_start_time_step, simulation_end_time_step = set_schema_simulation_period(schema, day_count, random_seed, root_directory,forced_simulation_start_time_step,forced_simulation_end_time_step)
    schema = set_active_observations(schema, active_observations)
    if Prints :
        print('Number of buildings:', building_count)
        print('Random Seed:', random_seed)
        print('Selected buildings:', buildings)
        print(f'Selected {day_count}-day period training time steps:',(simulation_start_time_step, simulation_end_time_step))
        print(f'Active observations:', active_observations)
    schema['central_agent'] = True

    return schema, buildings, simulation_start_time_step, simulation_end_time_step, active_observations, root_directory,random_seed, building_count, day_count

# Convert int32 σε int
def convert_to_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def add_datetime_column(schema, root_directory, output_directory):
    """Προσθέτει στήλη datetime στα δεδομένα """
    updated_schema = schema.copy()
    updated_buildings = {}

    for building_id, building_data in schema['buildings'].items():
        if not building_data.get('include', False):
            continue

        energy_simulation_path = Path(root_directory) / building_data['energy_simulation']
        energy_simulation_data = pd.read_csv(energy_simulation_path)

        # Εκτύπωση των πρώτων γραμμών του CSV για επιβεβαίωση των στηλών
        print("Αρχικά δεδομένα για το κτίριο:", building_id)
        print(energy_simulation_data.head())

        # Πρώτη ημέρα του dataset είναι 1η Αυγούστου 2016 στις 00:00
        start_date = datetime(2016, 8, 1, 0, 0)
        
        # Δημιουργία στήλης datetime με τη σωστή μετατροπή των ημερών και ωρών
        energy_simulation_data['datetime'] = [
            start_date + timedelta(days=int(day), hours=int(hour-1))
            for day, hour in zip(energy_simulation_data.index // 24, energy_simulation_data['Hour'])
        ]

        # Εκτύπωση των ενημερωμένων δεδομένων
        print(f"Δεδομένα για το κτίριο {building_id} μετά την προσθήκη της στήλης datetime:")
        print(energy_simulation_data.head())
        print(f"Αριθμός γραμμών: {len(energy_simulation_data)}\n")

        # Αποθήκευση των ενημερωμένων δεδομένων προσομοίωσης ενέργειας σε ξεχωριστό αρχείο
        output_path = Path(output_directory) / f"{building_id}_energy_simulation_data_with_datetime.csv"
        energy_simulation_data.to_csv(output_path, index=False)

        updated_buildings[building_id] = building_data
        updated_buildings[building_id]['energy_simulation_data_with_datetime_path'] = str(output_path)

    updated_schema['buildings'] = updated_buildings
    return updated_schema

