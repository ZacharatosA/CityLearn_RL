from imports import*

class CustomRewardFunction(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        r"""Initialize CustomRewardFunction.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        """

        super().__init__(env)

    def calculate(self) -> List[float]:
        r"""Returns reward for most recent action.


        Returns
        -------
        reward: List[float]
            Reward για τη μετάβαση στο τρέχον χρονικό βήμα.
        """

        reward_list = []

        for b in self.env.buildings:
            #Cost
            cost = b.net_electricity_consumption_cost[-1] 
            #Battery Info
            battery_capacity = b.electrical_storage.capacity_history[0] 
            battery_soc = b.electrical_storage.soc[-1]/battery_capacity 
            penalty = -(1.0 + np.sign(cost)*battery_soc) 
            #Reward for the Building
            reward = penalty*abs(cost) 
            reward_list.append(reward)
        reward = [sum(reward_list)]

        return reward
    
class CustomRewardFunctionWeighted(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        reward_list = []
        electricity_consumption_list = []

        #Weights
        Carbon_emissions_weight=2.5
        Cost_weight=5
        Electricity_consumption_weight=0.5


        for b in self.env.buildings:
            #Battery Info
            battery_capacity = b.electrical_storage.capacity_history[0]
            battery_soc = b.electrical_storage.soc[-1] / battery_capacity
            # Cost
            cost = b.net_electricity_consumption_cost[-1]
            penalty = -(1.0 + np.sign(cost) * battery_soc)
            cost_reward = Cost_weight * penalty * abs(cost)
            #---------------------------------------------------------------------
            # Υπολογισμός εκπομπή CO2 -----------
            carbon_emmission = b.net_electricity_consumption_emission[-1]
            carbon_emmission_reward = Carbon_emissions_weight * carbon_emmission
            
            electricity_consumption = b.net_electricity_consumption[-1]
            if battery_soc > 0.98:
              electricity_consumption = penalty * electricity_consumption
            electricity_consumption_reward = Electricity_consumption_weight * electricity_consumption
            electricity_consumption_list.append(electricity_consumption_reward)
            #-------------------------------------------------------------------------------
            reward_list.append(cost_reward)
            reward_list.append(carbon_emmission_reward)
        reward = [-abs(sum(electricity_consumption_list)) + sum(reward_list)]

        return reward

