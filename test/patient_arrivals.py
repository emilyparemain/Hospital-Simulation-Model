# Generate patient arrivals (as a CSV file)

# Patient groups should probably be changed as this is only split into short/medium/long stays
# Pts are also classified only by stay length, not ward type - amend to assign beds based on ward rules

import numpy as np
import random
import pandas as pd

# Parameters
SIMULATION_TIME = 365  # Simulate for 365 days
ARRIVAL_RATE = 0.78  # Patients per day (Poisson distribution)

# Split by gender
FEMALE_RATIO = 0.5  

# Create patient groups
PATIENT_GROUPS = {
    "Short": {"probability": 0.4, "mean_LoS": 3},  # Short stay: 3 days avg
    "Medium": {"probability": 0.4, "mean_LoS": 7},  # Medium stay: 7 days avg
    "Long": {"probability": 0.2, "mean_LoS": 12},  # Long stay: 12 days avg
}

def generate_patient_arrivals(simulation_days):
    # Generate patient arrivals (Poisson)
    np.random.seed(42)
    random.seed(42)

    patient_id = 1
    arrivals = []

    for day in range(simulation_days):
        # Number of arrivals for the current day
        num_arrivals = np.random.poisson(ARRIVAL_RATE)

        for _ in range(num_arrivals):
            # Assign gender based on probability
            gender = "Female" if random.random() < FEMALE_RATIO else "Male"

            # Assign patient group
            group_name = random.choices(
                list(PATIENT_GROUPS.keys()),
                weights=[PATIENT_GROUPS[g]["probability"] for g in PATIENT_GROUPS]
            )[0]

            # Sample Length of Stay (LoS) from an exponential distribution
            mean_LoS = PATIENT_GROUPS[group_name]["mean_LoS"]
            LoS = np.random.exponential(mean_LoS)

            # Store patient data
            arrivals.append({
                "Patient_ID": patient_id,
                "Day_Arrived": day,
                "Gender": gender,
                "Group": group_name,
                "Length_of_Stay": round(LoS, 1)
            })

            patient_id += 1

    return pd.DataFrame(arrivals)

# Generate data
df_patients = generate_patient_arrivals(SIMULATION_TIME)

print(df_patients.head())

# Save to CSV
df_patients.to_csv("patient_arrivals.csv", index = False)
print("\nPatient data saved to 'patient_arrivals.csv'.")
