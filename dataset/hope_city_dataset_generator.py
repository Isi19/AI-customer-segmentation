"""
Hope City Synthetic Household Energy Dataset Generator — Simple V2
==================================================================

Generates:
1) Household metadata table (same structure as original)
2) Hourly gross electricity consumption and gas consumption for a full calendar year

Key choice:
- electricity_kWh = gross household electricity consumption only
- gas_kWh = household gas consumption
"""

import os
import gzip
import random
import numpy as np
import pandas as pd

SEED = 42
YEAR = 2025
HOURS = 365 * 24
OUT_DIR = "."

ORIGINAL_METADATA = os.path.join(OUT_DIR, "hope_city_households_metadata.csv")
OUTPUT_METADATA = os.path.join(OUT_DIR, "hope_city_households_metadata_v2_same_as_original.csv")
OUTPUT_HOURLY = os.path.join(OUT_DIR, "hope_city_hourly_consumption_v2_2025.csv.gz")
OUTPUT_SAMPLE = os.path.join(OUT_DIR, "hope_city_hourly_consumption_v2_SAMPLE_100k.csv")


def base_daily(area, occ, occ_pat, children, income):
    base = 3.0 + 0.9 * occ + 0.012 * area

    if occ_pat == "retired":
        base *= 1.10
    elif occ_pat == "students":
        base *= 1.06
    elif occ_pat == "mixed":
        base *= 1.04

    if children == "yes":
        base *= 1.06

    base *= {"low": 0.96, "medium": 1.0, "high": 1.06}[income]
    base *= float(np.random.lognormal(0, 0.08))
    return float(base)


def generate_simple_v2():
    np.random.seed(SEED)
    random.seed(SEED)

    meta = pd.read_csv(ORIGINAL_METADATA)
    meta.to_csv(OUTPUT_METADATA, index=False)

    timestamps = pd.date_range(start=f"{YEAR}-01-01", periods=HOURS, freq="h")
    doy = timestamps.dayofyear.values
    hod = timestamps.hour.values
    dow = timestamps.dayofweek.values
    is_weekend = (dow >= 5).astype(int)

    season = np.sin(2 * np.pi * (doy - 80) / 365)
    temp = 11 + 10 * season + 3 * np.sin(2 * np.pi * (hod - 15) / 24) + np.random.normal(0, 1.3, size=HOURS)
    temp = np.clip(temp, -8, 35).astype(np.float32)

    ins_heat_factor = {"poor": 1.35, "medium": 1.0, "good": 0.72}
    prop_exposure_map = {
        "apartment": 0.72,
        "terraced": 0.92,
        "semi-detached": 1.00,
        "detached house": 1.12,
    }

    HDD = np.clip(19.0 - temp, 0, None).astype(np.float32)
    CDD = np.clip(temp - 22.5, 0, None).astype(np.float32)

    h24 = np.arange(24, dtype=np.float32)

    weekday_shape = (
        0.6 * np.exp(-0.5 * ((h24 - 8) / 2.0) ** 2)
        + 1.0 * np.exp(-0.5 * ((h24 - 20) / 2.5) ** 2)
        + 0.25
    ).astype(np.float32)
    weekday_shape /= weekday_shape.sum()

    weekend_shape = (
        0.55 * np.exp(-0.5 * ((h24 - 10) / 2.8) ** 2)
        + 0.85 * np.exp(-0.5 * ((h24 - 20) / 2.8) ** 2)
        + 0.35
    ).astype(np.float32)
    weekend_shape /= weekend_shape.sum()

    heat_profile = (
        0.75 * np.exp(-0.5 * ((hod - 7) / 3.0) ** 2)
        + 1.0 * np.exp(-0.5 * ((hod - 19) / 3.5) ** 2)
        + 0.55
    ).astype(np.float32)
    heat_profile /= heat_profile.mean()

    cool_profile = (
        0.9 * np.exp(-0.5 * ((hod - 16) / 4.0) ** 2)
        + 0.6 * np.exp(-0.5 * ((hod - 21) / 3.0) ** 2)
        + 0.15
    ).astype(np.float32)
    cool_profile /= cool_profile.mean()

    lighting_season = (1.0 + 0.08 * np.clip(-season, 0, None)).astype(np.float32)

    if os.path.exists(OUTPUT_HOURLY):
        os.remove(OUTPUT_HOURLY)

    with gzip.open(OUTPUT_HOURLY, "wt") as gz:
        gz.write("household_id,timestamp,electricity_kWh,gas_kWh\n")

        for row in meta.itertuples(index=False):
            hh = row.household_id
            area = row.floor_area_m2
            occ = row.num_occupants
            occ_pat = row.occupancy_pattern
            children = row.children_present
            income = row.income_band
            ins = row.insulation_quality
            heat = row.heating_type
            cool = row.cooling_system
            cook = row.cooking_fuel
            ev = row.ev_ownership
            prop = row.property_type

            daily_base = base_daily(area, occ, occ_pat, children, income)

            base_hourly = np.empty(HOURS, dtype=np.float32)
            for d in range(365):
                shape = weekend_shape if is_weekend[d * 24] else weekday_shape
                base_hourly[d * 24:(d + 1) * 24] = daily_base * lighting_season[d * 24] * shape

            cook_e = np.zeros(HOURS, dtype=np.float32)
            cook_g = np.zeros(HOURS, dtype=np.float32)
            cook_daily = (0.35 + 0.08 * occ) * (1.12 if children == "yes" else 1.0)
            cook_daily *= float(np.random.lognormal(0, 0.12))

            meal = np.zeros(24, dtype=np.float32)
            meal[7:9] += 0.25
            meal[18:21] += 0.55
            meal /= meal.sum()

            meal_we = meal.copy()
            meal_we[12:14] += 0.20
            meal_we /= meal_we.sum()

            for d in range(365):
                hrs = slice(d * 24, (d + 1) * 24)
                if cook == "electric":
                    cook_e[hrs] = cook_daily * (meal_we if is_weekend[d * 24] else meal)
                else:
                    cook_g[hrs] = cook_daily * (meal_we if is_weekend[d * 24] else meal)

            exposure = prop_exposure_map[prop]
            heat_thermal = (HDD * (area / 100.0) * ins_heat_factor[ins] * exposure * 0.55).astype(np.float32)
            heat_thermal *= float(np.random.lognormal(0, 0.10))
            heat_thermal *= heat_profile

            heat_e = np.zeros(HOURS, dtype=np.float32)
            heat_g = np.zeros(HOURS, dtype=np.float32)

            if heat == "gas boiler":
                eff = 0.88 + 0.05 * np.random.rand()
                heat_g = (heat_thermal / eff).astype(np.float32)
            elif heat == "electric heating":
                heat_e = heat_thermal
            elif heat == "heat pump":
                cop = np.clip(2.2 + 0.06 * temp, 2.0, 4.2).astype(np.float32)
                heat_e = (heat_thermal / cop).astype(np.float32)
            else:
                heat_e = (0.02 * heat_thermal).astype(np.float32)

            cool_e = np.zeros(HOURS, dtype=np.float32)
            if cool != "none":
                cool_thermal = (
                    CDD
                    * (area / 120.0)
                    * exposure
                    * (1.05 if ins == "poor" else 0.9 if ins == "good" else 1.0)
                    * 0.40
                ).astype(np.float32)
                cool_thermal *= float(np.random.lognormal(0, 0.15))
                cool_thermal *= cool_profile
                eer = (2.8 + 0.2 * np.random.rand()) if cool == "AC unit" else (3.2 + 0.3 * np.random.rand())
                cool_e = (cool_thermal / eer).astype(np.float32)

            ev_e = np.zeros(HOURS, dtype=np.float32)
            if ev == "yes":
                drive = np.random.normal(6.0, 1.8) + 0.6 * (occ - 2) + {"low": -0.5, "medium": 0.0, "high": 0.7}[income]
                drive = float(np.clip(drive, 2.5, 12.0)) * float(np.random.lognormal(0, 0.10))

                pattern = np.zeros(24, dtype=np.float32)
                pattern[19:24] = 0.62
                pattern[0:7] = 0.38 / 7.0
                pattern /= pattern.sum()
                pattern_we = np.roll(pattern, 1)

                for d in range(365):
                    hrs = slice(d * 24, (d + 1) * 24)
                    ev_e[hrs] = drive * (pattern_we if is_weekend[d * 24] else pattern)

            electricity = base_hourly + cook_e + heat_e + cool_e + ev_e
            gas = cook_g + heat_g

            noise = np.random.normal(0, 0.06, size=HOURS).astype(np.float32)
            noise = np.convolve(noise, np.array([0.15, 0.7, 0.15], dtype=np.float32), mode="same")

            electricity = np.clip(electricity * (1.0 + noise), 0, None).astype(np.float32)
            gas = np.clip(gas * (1.0 + 0.8 * noise), 0, None).astype(np.float32)

            df = pd.DataFrame({
                "household_id": np.full(HOURS, hh, dtype=np.int16),
                "timestamp": timestamps,
                "electricity_kWh": electricity,
                "gas_kWh": gas,
            })

            df.to_csv(gz, index=False, header=False, float_format="%.4f")

    with gzip.open(OUTPUT_HOURLY, "rt") as src, open(OUTPUT_SAMPLE, "wt") as dst:
        for i, line in enumerate(src):
            dst.write(line)
            if i >= 100000:
                break

    print("Done.")
    print("Metadata:", OUTPUT_METADATA)
    print("Hourly:", OUTPUT_HOURLY)
    print("Sample:", OUTPUT_SAMPLE)


if __name__ == "__main__":
    generate_simple_v2()

path = "/mnt/data/hope_city_dataset_generator_v2_simple.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print(path)
