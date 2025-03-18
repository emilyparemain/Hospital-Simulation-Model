# DES framework using SimPy

# Assuming homogeneity (all pts have same LoS); amend admit_patient() to use gender-based and group-based LoS distributions
# Link with other python script (patient_arrivals.py) - remove random.expovariate(ARRIVAL_RATE) and use data from CSV file generated from other script

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters
RANDOM_SEED = 42
SIM_TIME = 365  # Simulate 365 days
ARRIVAL_RATE = 0.78  # Average number of patients arriving per day (Poisson)
MEAN_LOS = 10  # Average length of stay in days (Exponential)

# Ward Configuration
TOTAL_BEDS = 3  # Total number of beds in the ward
MAX_WAIT_TIME = 2  # Max days a patient waits before leaving

# Initialise empty lists
waiting_times = []
occupancy_rates = []
patients_turned_away = 0

class Hospital:
    # Hospital ward simulation with a fixed number of beds
    def __init__(self, env, total_beds):
        self.env = env
        self.beds = simpy.Resource(env, capacity=total_beds)

    def admit_patient(self, patient_id):
        # Patient admission process
        arrival_time = self.env.now
        with self.beds.request() as req:
            # Patient waits for bed
            results = yield req | self.env.timeout(MAX_WAIT_TIME)
            wait_time = self.env.now - arrival_time

            if req in results:  # Bed becomes available
                waiting_times.append(wait_time)
                los = np.random.exponential(MEAN_LOS)  # LoS
                yield self.env.timeout(los)  # Patient stays in the ward

                occupancy_rates.append(len(self.beds.users) / TOTAL_BEDS)
            else:
                global patients_turned_away
                patients_turned_away += 1  # Patient leaves after waiting too long

def patient_generator(env, hospital):
    # Generate patient arrivals at Poisson-distributed intervals
    patient_id = 0
    while True:
        yield env.timeout(random.expovariate(ARRIVAL_RATE))  # Poisson arrivals
        env.process(hospital.admit_patient(patient_id))  # Process patient admission
        patient_id += 1

random.seed(RANDOM_SEED)
env = simpy.Environment()
hospital = Hospital(env, TOTAL_BEDS)

# Start processes
env.process(patient_generator(env, hospital))

# Run
env.run(until = SIM_TIME)

# Results Analysis
print(f"\nSimulation Results over {SIM_TIME} days:")
print(f"- Average Waiting Time: {np.mean(waiting_times):.2f} days")
print(f"- Bed Occupancy Rate: {np.mean(occupancy_rates) * 100:.1f}%")
print(f"- Patients Turned Away: {patients_turned_away}")

# Plot
plt.hist(waiting_times, bins = 20, edgecolor = 'black', alpha = 0.7)
plt.xlabel("Waiting Time (Days)")
plt.ylabel("Number of Patients")
plt.title("Distribution of Patient Waiting Times")
plt.show()

