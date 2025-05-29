import pandas as pd
import os

# Get the current directory of the script
script_dir = os.path.dirname(__file__)

# Define paths to the main normalized CSV files
traffic_influx_file = os.path.join(script_dir, "../results", "candidate_locations_with_normalized_traffic_influx.csv")
crash_risk_file = os.path.join(script_dir, "../results/accident_data_analysis", "candidate_locations_normalized_crash_scores.csv")
detour_time_file = os.path.join(script_dir, "../results", "normalized_detour_times.csv")
capacity_file = os.path.join(script_dir, "../results", "normalized_truck_capacity.csv")

print("="*80)
print("DEBUGGING MISSING LOCATION IN MERGE")
print("="*80)

# Load each dataset
traffic_influx_df = pd.read_csv(traffic_influx_file)
crash_risk_df = pd.read_csv(crash_risk_file)
detour_time_df = pd.read_csv(detour_time_file)
capacity_df = pd.read_csv(capacity_file)

print(f"Initial counts:")
print(f"- Traffic influx: {len(traffic_influx_df)} records")
print(f"- Crash risk: {len(crash_risk_df)} records")
print(f"- Detour time: {len(detour_time_df)} records")
print(f"- Capacity: {len(capacity_df)} records")

# Create location_id for each dataset
def create_location_id_debug(df, name):
    """Create location_id and track changes"""
    if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
        df['location_id'] = (df['centroid_x'].round(6).astype(str) + "_" + 
                            df['centroid_y'].round(6).astype(str))
    else:
        df['location_id'] = df['FID'].astype(str)
    
    print(f"\n{name} after creating location_id:")
    print(f"  Total records: {len(df)}")
    print(f"  Unique location_ids: {df['location_id'].nunique()}")
    
    # Remove exact duplicates
    df_no_exact_dups = df.drop_duplicates()
    if len(df_no_exact_dups) != len(df):
        print(f"  Removed {len(df) - len(df_no_exact_dups)} exact duplicates")
    
    # Remove location_id duplicates
    df_clean = df_no_exact_dups.drop_duplicates(subset=['location_id'], keep='first')
    if len(df_clean) != len(df_no_exact_dups):
        print(f"  Removed {len(df_no_exact_dups) - len(df_clean)} location_id duplicates")
        
        # Show which location_ids were duplicated
        duplicated_locations = df_no_exact_dups[df_no_exact_dups.duplicated(subset=['location_id'], keep=False)]
        if len(duplicated_locations) > 0:
            print(f"  Duplicated location_ids:")
            for location_id in duplicated_locations['location_id'].unique():
                dup_data = duplicated_locations[duplicated_locations['location_id'] == location_id]
                print(f"    {location_id}:")
                for _, row in dup_data.iterrows():
                    print(f"      FID:{row['FID']}, Name:{row['ComplexNam']}")
    
    print(f"  Final unique records: {len(df_clean)}")
    return df_clean

# Clean each dataset and track location_ids
traffic_influx_clean = create_location_id_debug(traffic_influx_df.copy(), "Traffic Influx")
crash_risk_clean = create_location_id_debug(crash_risk_df.copy(), "Crash Risk")
detour_time_clean = create_location_id_debug(detour_time_df.copy(), "Detour Time")
capacity_clean = create_location_id_debug(capacity_df.copy(), "Capacity")

# Get sets of location_ids from each dataset
traffic_ids = set(traffic_influx_clean['location_id'])
crash_ids = set(crash_risk_clean['location_id'])
detour_ids = set(detour_time_clean['location_id'])
capacity_ids = set(capacity_clean['location_id'])

print(f"\n" + "="*60)
print("LOCATION_ID SET ANALYSIS")
print("="*60)
print(f"Traffic influx location_ids: {len(traffic_ids)}")
print(f"Crash risk location_ids: {len(crash_ids)}")
print(f"Detour time location_ids: {len(detour_ids)}")
print(f"Capacity location_ids: {len(capacity_ids)}")

# Find the intersection (common to all datasets)
common_ids = traffic_ids & crash_ids & detour_ids & capacity_ids
print(f"\nLocation_ids present in ALL datasets: {len(common_ids)}")

# Find which location_ids are missing from each dataset
missing_from_crash = traffic_ids - crash_ids
missing_from_detour = traffic_ids - detour_ids
missing_from_capacity = traffic_ids - capacity_ids

missing_from_traffic = crash_ids - traffic_ids
missing_from_traffic.update(detour_ids - traffic_ids)
missing_from_traffic.update(capacity_ids - traffic_ids)

print(f"\n" + "="*60)
print("MISSING LOCATION_IDS ANALYSIS")
print("="*60)

if missing_from_crash:
    print(f"\nLocation_ids in Traffic but missing from Crash Risk ({len(missing_from_crash)}):")
    for location_id in missing_from_crash:
        traffic_row = traffic_influx_clean[traffic_influx_clean['location_id'] == location_id].iloc[0]
        print(f"  {location_id} -> FID:{traffic_row['FID']}, Name:{traffic_row['ComplexNam']}")

if missing_from_detour:
    print(f"\nLocation_ids in Traffic but missing from Detour Time ({len(missing_from_detour)}):")
    for location_id in missing_from_detour:
        traffic_row = traffic_influx_clean[traffic_influx_clean['location_id'] == location_id].iloc[0]
        print(f"  {location_id} -> FID:{traffic_row['FID']}, Name:{traffic_row['ComplexNam']}")

if missing_from_capacity:
    print(f"\nLocation_ids in Traffic but missing from Capacity ({len(missing_from_capacity)}):")
    for location_id in missing_from_capacity:
        traffic_row = traffic_influx_clean[traffic_influx_clean['location_id'] == location_id].iloc[0]
        print(f"  {location_id} -> FID:{traffic_row['FID']}, Name:{traffic_row['ComplexNam']}")

if missing_from_traffic:
    print(f"\nLocation_ids in other datasets but missing from Traffic ({len(missing_from_traffic)}):")
    for location_id in missing_from_traffic:
        # Find which dataset has this location_id
        if location_id in crash_ids:
            row = crash_risk_clean[crash_risk_clean['location_id'] == location_id].iloc[0]
            print(f"  {location_id} -> FID:{row['FID']}, Name:{row['ComplexNam']} (from Crash Risk)")
        elif location_id in detour_ids:
            row = detour_time_clean[detour_time_clean['location_id'] == location_id].iloc[0]
            print(f"  {location_id} -> FID:{row['FID']}, Name:{row['ComplexNam']} (from Detour Time)")
        elif location_id in capacity_ids:
            row = capacity_clean[capacity_clean['location_id'] == location_id].iloc[0]
            print(f"  {location_id} -> FID:{row['FID']}, Name:{row['ComplexNam']} (from Capacity)")

# Check for coordinate precision issues
print(f"\n" + "="*60)
print("COORDINATE PRECISION CHECK")
print("="*60)

# Look for very similar coordinates that might be creating different location_ids
def check_coordinate_precision(df1, df2, name1, name2):
    """Check for locations that might be the same but with different coordinate precision"""
    print(f"\nComparing {name1} vs {name2}:")
    
    # Get locations that are in df1 but not in df2
    df1_ids = set(df1['location_id'])
    df2_ids = set(df2['location_id'])
    
    only_in_df1 = df1_ids - df2_ids
    only_in_df2 = df2_ids - df1_ids
    
    if only_in_df1 and only_in_df2:
        print(f"  {len(only_in_df1)} location_ids only in {name1}")
        print(f"  {len(only_in_df2)} location_ids only in {name2}")
        
        # Check if any coordinates are very close
        for id1 in only_in_df1:
            row1 = df1[df1['location_id'] == id1].iloc[0]
            x1, y1 = row1['centroid_x'], row1['centroid_y']
            
            for id2 in only_in_df2:
                row2 = df2[df2['location_id'] == id2].iloc[0]
                x2, y2 = row2['centroid_x'], row2['centroid_y']
                
                # Check if coordinates are very close (within 0.000001 degrees)
                if abs(x1 - x2) < 0.000001 and abs(y1 - y2) < 0.000001:
                    print(f"    POTENTIAL MATCH:")
                    print(f"      {name1}: {id1} -> FID:{row1['FID']}, Name:{row1['ComplexNam']}, Coords:({x1:.8f}, {y1:.8f})")
                    print(f"      {name2}: {id2} -> FID:{row2['FID']}, Name:{row2['ComplexNam']}, Coords:({x2:.8f}, {y2:.8f})")

# Check each pair of datasets for coordinate precision issues
check_coordinate_precision(traffic_influx_clean, crash_risk_clean, "Traffic Influx", "Crash Risk")
check_coordinate_precision(traffic_influx_clean, detour_time_clean, "Traffic Influx", "Detour Time")
check_coordinate_precision(traffic_influx_clean, capacity_clean, "Traffic Influx", "Capacity")

print(f"\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("To fix the missing location issue:")
print("1. Check the original data files for the missing location")
print("2. Verify if it's a coordinate precision issue")
print("3. Consider using 'outer' join instead of 'inner' join to include all locations")
print("4. Fill missing values with appropriate defaults")
print("="*60)