"""
Dataset Generator for Project 14: Airline Flight Delay & On-Time Performance Analysis
Generates a realistic synthetic flight-delays dataset: '14_flight_delays.csv'
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_flight_data(num_records=6000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    airlines = [
        "Delta Air Lines",
        "American Airlines",
        "United Airlines",
        "Southwest Airlines",
        "JetBlue Airways",
        "SkyWest Airlines"
    ]
    airline_weights = [0.24, 0.23, 0.21, 0.18, 0.08, 0.06]

    airports = [
        "ATL", "ORD", "DFW", "DEN", "CLT",
        "LAX", "JFK", "SFO", "SEA", "LAS",
        "MIA", "BOS", "EWR", "PHX"
    ]
    # Congestion weights for origins (EWR, ORD, SFO, JFK have higher delay propensities)
    airport_delay_bias = {
        "EWR": 14.5, "ORD": 12.0, "JFK": 11.5, "SFO": 10.0,
        "DFW": 6.0, "DEN": 5.5, "BOS": 5.0, "LAX": 4.5,
        "MIA": 4.0, "ATL": 3.0, "SEA": 2.5, "CLT": 2.0,
        "PHX": 1.0, "LAS": 1.5
    }

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    date_range_days = (end_date - start_date).days

    records = []

    hour_weights = np.array([
        0.01, 0.01, 0.005, 0.005, 0.01, 0.03, # 00-05 Night/Early morning
        0.06, 0.07, 0.08, 0.07, 0.06, 0.06,   # 06-11 Morning
        0.06, 0.06, 0.06, 0.07, 0.07, 0.07,   # 12-17 Afternoon/Evening
        0.07, 0.06, 0.04, 0.03, 0.02, 0.01    # 18-23 Night
    ])
    hour_weights = hour_weights / hour_weights.sum()

    for i in range(1, num_records + 1):
        flight_id = f"FL-{10000 + i}"
        
        # Flight date
        random_days = random.randint(0, date_range_days)
        flight_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # Airline
        airline = np.random.choice(airlines, p=airline_weights)
        
        # Origin & Destination (cannot be the same)
        origin = random.choice(airports)
        dest_candidates = [a for a in airports if a != origin]
        destination = random.choice(dest_candidates)
        
        # Scheduled Departure Time (HH:MM)
        hour = int(np.random.choice(range(24), p=hour_weights))
        minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        sched_dep_time = f"{hour:02d}:{minute:02d}"

        # Cancellation (~2.2%) & Diversion (~0.8%)
        is_cancelled = 1 if np.random.random() < 0.022 else 0
        is_diverted = 1 if (is_cancelled == 0 and np.random.random() < 0.008) else 0

        # Delay simulation
        # Base delay distribution: mostly on-time/early, some moderate, some severe outliers
        bias = airport_delay_bias.get(origin, 0.0)
        # Evening flights accumulate cascading delay
        time_bias = 6.0 if (17 <= hour <= 21) else (3.0 if (12 <= hour <= 16) else -2.0)
        
        prob_delay_type = np.random.random()
        if prob_delay_type < 0.65:
            # On-time or early departure
            dep_delay = int(np.random.normal(loc=-2 + (bias * 0.1), scale=5))
            if dep_delay > 15:
                dep_delay = 14
        elif prob_delay_type < 0.85:
            # Minor to moderate delay
            dep_delay = int(np.random.exponential(scale=18) + 10 + bias * 0.4 + time_bias)
        elif prob_delay_type < 0.96:
            # Moderate to heavy delay
            dep_delay = int(np.random.exponential(scale=35) + 35 + bias * 0.7 + time_bias)
        else:
            # Severe delay outlier (up to 300+ minutes)
            dep_delay = int(np.random.exponential(scale=65) + 85 + bias + time_bias)

        # Bounds check
        if dep_delay < -25:
            dep_delay = -25

        # Arrival delay correlated with departure delay (pilots adjust speed, hold patterns)
        enroute_adjustment = int(np.random.normal(loc=-3, scale=7))
        arr_delay = dep_delay + enroute_adjustment

        # Delay cause assignment
        if is_cancelled:
            arr_delay = np.nan
            dep_delay = np.nan if np.random.random() < 0.5 else dep_delay
            delay_cause = np.random.choice(["Weather", "Carrier", "Air Traffic"], p=[0.55, 0.30, 0.15])
        elif is_diverted:
            arr_delay = np.nan
            delay_cause = np.random.choice(["Weather", "Air Traffic"], p=[0.70, 0.30])
        elif dep_delay > 15 or arr_delay > 15:
            delay_cause = np.random.choice(
                ["Air Traffic", "Carrier", "Weather"],
                p=[0.44, 0.34, 0.22]
            )
        else:
            delay_cause = "None"

        # Introduce a few random missing arrival delays for sensor missingness (~1.2%)
        if not is_cancelled and not is_diverted and np.random.random() < 0.012:
            arr_delay = np.nan

        records.append({
            "Flight_ID": flight_id,
            "Flight_Date": flight_date,
            "Airline": airline,
            "Origin": origin,
            "Destination": destination,
            "Scheduled_Departure_Time": sched_dep_time,
            "Departure_Delay_Mins": dep_delay,
            "Arrival_Delay_Mins": arr_delay,
            "Delay_Cause": delay_cause,
            "Cancelled": is_cancelled,
            "Diverted": is_diverted
        })

    df = pd.DataFrame(records)

    # Introduce duplicate Flight_ID entries (~50 duplicates)
    dup_indices = np.random.choice(len(df), size=50, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    # Append duplicates
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle the dataset so duplicates appear naturally
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_flight_data(num_records=6000, seed=42)
    output_path = r"C:\Users\Naitik Singh\.gemini\antigravity\scratch\Airline_Flight_Delay_Analysis\14_flight_delays.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated dataset successfully saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))
    print(df.head())
