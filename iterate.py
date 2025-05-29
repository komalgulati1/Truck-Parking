import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Define constants and parameters at the very beginning of the file
MAX_BUDGET = 175e6  # Maximum budget of interest (175 million)
COST_NO_SERVICE_MIN = 800000  # $800,000 for no service facilities (minimum)
COST_NO_SERVICE_MAX = 1.1e6  # $1.1 million for no service facilities (maximum)
COST_FULL_SERVICE_MIN = 11e6  # $11 million for full-service facilities (minimum)
COST_FULL_SERVICE_MAX = 14e6  # $14 million for full-service facilities (maximum)

# Define excluded facility types
EXCLUDED_FACILITY_TYPES = ["DMV", "License", "Welcome Center", "Municipal", "Courthouse", "Rest Area"]

# Weight for each component in the composite score
WEIGHTS = {
    'traffic_influx': 0.25,
    'crash_risk': 0.25,
    'accessibility': 0.25,
    'capacity': 0.25
}

# Get the current directory of the script
script_dir = os.path.dirname(__file__)
if not script_dir:
    script_dir = '.'

def load_data():
    """
    Load and prepare the candidate locations and existing facilities data
    """
    print("Loading and preparing data...")
    
    # Load candidate locations with composite scores
    candidate_path = os.path.join(script_dir, "../results/composite_prioritization_scores.csv")
    candidates_df = pd.read_csv(candidate_path)
    # REMOVE DUPLICATES HERE
    candidates_df = candidates_df.drop_duplicates(subset=['ComplexNam'])
    print(f"Removed duplicates by ComplexNam, remaining: {len(candidates_df)} candidates")
    
    # Load existing facilities
    existing_path = os.path.join(script_dir, "../datasets/existing_fac_without_weigh_station.shp")
    
    try:
        # Try to load as a CSV first
        existing_facilities = pd.read_csv(existing_path)
        # Convert to GeoDataFrame if it's a CSV
        if 'geometry' not in existing_facilities.columns and 'Longitude' in existing_facilities.columns and 'Latitude' in existing_facilities.columns:
            geometry = [Point(lon, lat) for lon, lat in zip(existing_facilities['Longitude'], existing_facilities['Latitude'])]
            existing_facilities = gpd.GeoDataFrame(existing_facilities, geometry=geometry, crs="EPSG:4326")
    except:
        # If not a CSV, try to load as a shapefile
        existing_facilities = gpd.read_file(existing_path)
    
    print(f"Loaded {len(candidates_df)} candidate locations and {len(existing_facilities)} existing facilities")
    
    # Convert coordinates to proper geometry for candidate locations
    if 'geometry' not in candidates_df.columns and 'centroid_x' in candidates_df.columns and 'centroid_y' in candidates_df.columns:
        geometry = [Point(x, y) for x, y in zip(candidates_df['centroid_x'], candidates_df['centroid_y'])]
        candidates_gdf = gpd.GeoDataFrame(candidates_df, geometry=geometry, crs="EPSG:4326")
    else:
        candidates_gdf = gpd.GeoDataFrame(candidates_df, geometry='geometry', crs="EPSG:4326")
    
    # Ensure both datasets have the same CRS
    if existing_facilities.crs != candidates_gdf.crs:
        existing_facilities = existing_facilities.to_crs(candidates_gdf.crs)
    
    # Filter out facilities with missing names or where the name is "0"
    initial_count = len(candidates_gdf)
    candidates_gdf = candidates_gdf[
        candidates_gdf['ComplexNam'].notna() & 
        (candidates_gdf['ComplexNam'] != '') &
        (candidates_gdf['ComplexNam'] != '0')
    ]
    filtered_count = initial_count - len(candidates_gdf)
    print(f"Filtered out {filtered_count} facilities with missing, empty, or '0' names")
    print(f"Remaining candidate locations: {len(candidates_gdf)}")
    
    # Filter out DMV sites and other inappropriate facility types
    initial_count = len(candidates_gdf)
    filtered_candidates = candidates_gdf.copy()
    for excluded_type in EXCLUDED_FACILITY_TYPES:
        filtered_candidates = filtered_candidates[~filtered_candidates['ComplexNam'].str.contains(excluded_type, case=False, na=False)]
    
    candidates_gdf = filtered_candidates
    excluded_count = initial_count - len(candidates_gdf)
    print(f"Filtered out {excluded_count} inappropriate facility types (DMV, License, etc.)")
    print(f"Remaining candidate locations after type filtering: {len(candidates_gdf)}")
    
    # Calculate distance from each candidate to existing facilities
    candidates_gdf = calculate_proximity_to_existing(candidates_gdf, existing_facilities)
    
    # Calculate the development cost for each candidate location
    candidates_gdf = calculate_development_costs(candidates_gdf)
    
    return candidates_gdf, existing_facilities

def calculate_proximity_to_existing(candidates_gdf, existing_facilities, facility_type='no_service'):
    """
    Calculate the distance from each candidate to all existing facilities,
    then calculate the average distance to the nearest 3 facilities.
    """
    print(f"Calculating proximity to existing facilities for {facility_type} scenario...")
    
    # Check if the dataframe is empty
    if len(candidates_gdf) == 0:
        print("No candidates remaining to calculate proximity.")
        return candidates_gdf
    
    # Initialize columns to store distances
    distances = []
    
    # For each candidate location
    for idx, candidate in candidates_gdf.iterrows():
        # Calculate distance to all existing facilities
        if len(existing_facilities) > 0:
            dists = [candidate.geometry.distance(facility.geometry) for _, facility in existing_facilities.iterrows()]
            
            # Sort distances and take the average of the nearest 3 (or fewer if there are less than 3)
            nearest_n = min(3, len(dists))
            if nearest_n > 0:
                avg_nearest_3 = np.mean(sorted(dists)[:nearest_n])
            else:
                avg_nearest_3 = float('inf')  # No existing facilities
        else:
            avg_nearest_3 = float('inf')  # No existing facilities
        
        distances.append(avg_nearest_3)
    
    # Add the distances to the dataframe
    candidates_gdf['average_distance_nearest_3'] = distances
    
    # Calculate proximity score (normalized to 0-1)
    max_dist = candidates_gdf['average_distance_nearest_3'].max()
    min_dist = candidates_gdf['average_distance_nearest_3'].min()
    
    if max_dist > min_dist:
        if facility_type == 'no_service':
            # For no-service: CLOSER is BETTER (inverse relationship - farther = lower score)
            candidates_gdf['proximity_score'] = (1 - (
                (candidates_gdf['average_distance_nearest_3'] - min_dist) / 
                (max_dist - min_dist)
            )) ** 2  # Square to make the effect more pronounced
        else:  # full_service
            # For full-service: FARTHER is BETTER (direct relationship - farther = higher score)
            candidates_gdf['proximity_score'] = (
                (candidates_gdf['average_distance_nearest_3'] - min_dist) / 
                (max_dist - min_dist)
            ) ** 2  # Square to make the effect more pronounced
    else:
        candidates_gdf['proximity_score'] = 1.0  # All have the same distance
    
    return candidates_gdf

def calculate_development_costs(candidates_gdf):
    """
    Calculate the development cost for each candidate location based on capacity_value.
    Uses 'capacity_value' field directly from the loaded CSV data.
    """
    print("Calculating development costs...")
    
    # Use capacity_value directly from the CSV
    if 'capacity_value' not in candidates_gdf.columns:
        raise ValueError("'capacity_value' field not found in the data. Please ensure the CSV contains this field.")
    
    capacity_column = 'capacity_value'
    print("Using 'capacity_value' field from CSV for cost calculations")
    
    # Cost model constants
    SITE_PREP_NO_SERVICE = 200000  # $200,000 for basic site preparation
    COST_PER_SPACE_NO_SERVICE = 10000  # $10,000 per space for no-service
    
    SITE_PREP_FULL_SERVICE = 3000000  # $3 million for full-service site preparation
    COST_PER_SPACE_FULL_SERVICE = 100000  # $100,000 per space for full-service
    
    # Calculate costs using capacity_value
    print(f"Calculating costs based on {capacity_column} field")
    print(f"Capacity range: {candidates_gdf[capacity_column].min()} to {candidates_gdf[capacity_column].max()} spaces")
    
    candidates_gdf['cost_no_service'] = SITE_PREP_NO_SERVICE + (candidates_gdf[capacity_column] * COST_PER_SPACE_NO_SERVICE)
    candidates_gdf['cost_full_service'] = SITE_PREP_FULL_SERVICE + (candidates_gdf[capacity_column] * COST_PER_SPACE_FULL_SERVICE)
    
    # DO NOT clip costs - let them vary based on actual capacity
    # Each facility's cost should be proportional to its capacity
    print(f"No-service costs calculated (no clipping applied)")
    print(f"  Range: ${candidates_gdf['cost_no_service'].min():,.0f} to ${candidates_gdf['cost_no_service'].max():,.0f}")
    print(f"Full-service costs calculated (no clipping applied)")
    print(f"  Range: ${candidates_gdf['cost_full_service'].min():,.0f} to ${candidates_gdf['cost_full_service'].max():,.0f}")
    
    # Store the capacity value used for reference in later functions
    candidates_gdf['capacity_used_for_costing'] = candidates_gdf[capacity_column]
    
    print(f"Cost calculation completed using {capacity_column}")
    print(f"Costs now vary by facility capacity:")
    print(f"  No-service: ${candidates_gdf['cost_no_service'].min()/1e6:.3f}M to ${candidates_gdf['cost_no_service'].max()/1e6:.3f}M")
    print(f"  Full-service: ${candidates_gdf['cost_full_service'].min()/1e6:.3f}M to ${candidates_gdf['cost_full_service'].max()/1e6:.3f}M")
    
    return candidates_gdf

def update_composite_score(candidates_gdf, facility_type='no_service'):
    """
    Update the composite score considering the proximity to existing facilities
    """
    print(f"Updating composite scores for {facility_type} scenario...")
    
    # Check if the dataframe is empty
    if len(candidates_gdf) == 0:
        print("No candidates remaining to score.")
        return candidates_gdf
    
    # Increase proximity weight for clearer differentiation between strategies
    proximity_weight = 0.4  # Increased from 0.2 to 0.4 for stronger effect
    
    # Update weights to include proximity (reduce original weights proportionally)
    factor = (1 - proximity_weight) / sum(WEIGHTS.values())
    updated_weights = {
        'traffic_influx_norm': WEIGHTS['traffic_influx'] * factor,
        'crash_risk_norm': WEIGHTS['crash_risk'] * factor,
        'accessibility_norm': WEIGHTS['accessibility'] * factor,
        'capacity_norm': WEIGHTS['capacity'] * factor,
        'proximity_score': proximity_weight  # Higher weight for proximity
    }
    
    # Calculate updated composite score
    candidates_gdf['updated_composite_score'] = (
        candidates_gdf['traffic_influx_norm'] * updated_weights['traffic_influx_norm'] +
        candidates_gdf['crash_risk_norm'] * updated_weights['crash_risk_norm'] +
        candidates_gdf['accessibility_norm'] * updated_weights['accessibility_norm'] +
        candidates_gdf['capacity_norm'] * updated_weights['capacity_norm'] +
        candidates_gdf['proximity_score'] * updated_weights['proximity_score']
    )
    
    return candidates_gdf

def infinite_budget_allocation(candidates_gdf, existing_facilities, cost_column, output_prefix, max_budget=MAX_BUDGET):
    """
    Iteratively select locations based on composite score until reaching maximum budget,
    tracking the order of selection and cumulative budget for each facility
    """
    print(f"\nStarting continuous budget allocation for {output_prefix} scenario up to ${max_budget/1e6:.1f}M")
    
    # Make a copy of the dataframes
    candidates_working = candidates_gdf.copy()
    existing_working = existing_facilities.copy()
    
    # Initialize tracking
    selection_order = []
    cumulative_budget = 0
    cumulative_capacity = 0
    
    # Set proximity strategy based on facility type
    facility_type = output_prefix
    
    # Create folders for saving results
    base_output_folder = os.path.join(script_dir, "../results/iterative_results")
    os.makedirs(base_output_folder, exist_ok=True)
    
    selection_folder = os.path.join(base_output_folder, f"{output_prefix}_continuous")
    os.makedirs(selection_folder, exist_ok=True)
    
    # Initial proximity calculation with appropriate strategy
    candidates_working = calculate_proximity_to_existing(candidates_working, existing_working, facility_type)
    candidates_working = update_composite_score(candidates_working, facility_type)
    
    # Loop until reaching maximum budget or running out of candidates
    selection_count = 0
    
    # Create a folder for iteration results
    iteration_folder = os.path.join(script_dir, f"../results/iterations/{output_prefix}")
    os.makedirs(iteration_folder, exist_ok=True)
    
    while len(candidates_working) > 0:
        # Sort candidates by updated composite score
        candidates_working = candidates_working.sort_values('updated_composite_score', ascending=False)
        
        # Select the highest-ranked candidate
        selected = candidates_working.iloc[0]
        
        # Skip facilities with name value "0"
        facility_name = selected.get('ComplexNam', '')
        if facility_name == '0':
            print(f"Skipping facility with name value '0' (FID {selected['FID']})")
            candidates_working = candidates_working.iloc[1:]
            continue
        
        # Check if adding this facility would exceed the maximum budget
        new_total_budget = cumulative_budget + selected[cost_column]
        if new_total_budget > max_budget:
            print(f"Maximum budget of ${max_budget/1e6:.1f}M would be exceeded. Stopping selection process.")
            break
        
        # Add facility
        selection_count += 1
        cumulative_budget = new_total_budget
        
        # Use the capacity that was used for costing (stored during cost calculation)
        current_capacity = selected.get('capacity_used_for_costing', 0)
        cumulative_capacity += current_capacity
        
        # Extract centroid coordinates
        if hasattr(selected.geometry, 'centroid'):
            centroid_x = selected.geometry.centroid.x
            centroid_y = selected.geometry.centroid.y
        else:
            # Try to use existing centroid_x and centroid_y if available
            centroid_x = selected.get('centroid_x', 0)
            centroid_y = selected.get('centroid_y', 0)
        
        # Add to selection order with tracking information
        selection_order.append({
            'Selection_Order': selection_count,
            'FID': selected['FID'],
            'Facility_Name': selected.get('ComplexNam', 'Unnamed facility'),
            'Cost': selected[cost_column],
            'Capacity': current_capacity,
            'Cumulative_Budget': cumulative_budget,
            'Cumulative_Capacity': cumulative_capacity,
            'Composite_Score': selected['updated_composite_score'],
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'geometry': selected.geometry  # Preserve geometry for spatial joins later
        })
        
        # Print iteration results
        print(f"Iteration #{selection_count}: Selected {selected.get('ComplexNam', 'Unnamed facility')} (FID {selected['FID']})")
        print(f"  Cost: ${selected[cost_column]/1e6:.3f}M, Cumulative: ${cumulative_budget/1e6:.3f}M")
        print(f"  Capacity: {current_capacity}, Cumulative: {cumulative_capacity}")
        print(f"  Location: ({centroid_x:.6f}, {centroid_y:.6f})")
        print(f"  Composite Score: {selected['updated_composite_score']:.4f}")
        
        # Save iteration result to CSV
        iteration_df = pd.DataFrame([{
            'Iteration': selection_count,
            'FID': selected['FID'],
            'Facility_Name': selected.get('ComplexNam', 'Unnamed facility'),
            'Cost_Millions': selected[cost_column] / 1e6,
            'Capacity': current_capacity,
            'Cumulative_Budget_Millions': cumulative_budget / 1e6,
            'Cumulative_Capacity': cumulative_capacity,
            'Composite_Score': selected['updated_composite_score'],
            'centroid_x': centroid_x,
            'centroid_y': centroid_y
        }])
        
        iteration_file = os.path.join(iteration_folder, f"iteration_{selection_count:03d}.csv")
        iteration_df.to_csv(iteration_file, index=False, float_format='%.6f')
        
        # Remove selected location from candidates
        candidates_working = candidates_working[candidates_working['FID'] != selected['FID']]
        
        # Add selected location to existing facilities
        selected_gdf = gpd.GeoDataFrame([selected], geometry='geometry', crs=existing_working.crs)
        existing_working = pd.concat([existing_working, selected_gdf], ignore_index=True)
        
        # Recalculate distances and update scores
        candidates_working = calculate_proximity_to_existing(candidates_working, existing_working, facility_type)
        candidates_working = update_composite_score(candidates_working, facility_type)
    
    # Create a DataFrame with selection order information
    selection_df = pd.DataFrame(selection_order)
    
    # Convert to GeoDataFrame to preserve geometry for spatial joins
    if 'geometry' in selection_df.columns:
        selection_gdf = gpd.GeoDataFrame(selection_df, geometry='geometry', crs=candidates_gdf.crs)
    else:
        selection_gdf = selection_df
    
    # Save selection order to CSV
    output_folder = os.path.join(script_dir, "../results/selection_order")
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, f"{output_prefix}_selection_order.csv")
    # Drop geometry column before saving to CSV
    selection_csv = selection_df.copy()
    if 'geometry' in selection_csv.columns:
        selection_csv = selection_csv.drop(columns=['geometry'])
    selection_csv.to_csv(output_file, index=False)
    
    # Save a human-readable version with millions and coordinates
    output_file_readable = os.path.join(output_folder, f"{output_prefix}_selection_order_readable.csv")
    readable_df = selection_csv.copy()
    readable_df['Cost_Millions'] = readable_df['Cost'] / 1e6
    readable_df['Cumulative_Budget_Millions'] = readable_df['Cumulative_Budget'] / 1e6
    readable_df = readable_df[['Selection_Order', 'FID', 'Facility_Name', 'Cost_Millions', 
                              'Cumulative_Budget_Millions', 'Capacity', 'Cumulative_Capacity',
                              'centroid_x', 'centroid_y', 'Composite_Score']]
    readable_df.to_csv(output_file_readable, index=False, float_format='%.6f')
    
    print(f"\nResults for {output_prefix} scenario:")
    print(f"Total selected locations: {len(selection_df)}")
    print(f"Total capacity: {cumulative_capacity} spaces")
    print(f"Total cost: ${cumulative_budget/1e6:.3f}M")
    print(f"Detailed selection results saved to {output_file_readable}")
    print(f"Individual iteration results saved to {iteration_folder}")
    
    return selection_gdf

def create_true_staircase_plot(selection_df, color, label):
    """
    Given a facility selection dataframe, create x, y for a pure staircase step plot
    where steps are always flat (horizontal), only rising vertically at a facility addition.
    """
    budgets = list(selection_df['Cumulative_Budget'] / 1e6)
    counts = list(np.arange(1, len(selection_df) + 1))
    x = [0]
    y = [0]
    for i in range(len(budgets)):
        # Extend horizontal line at current count up to the new budget
        x.append(budgets[i])
        y.append(y[-1])
        # Vertical jump at this budget to the new facility count
        x.append(budgets[i])
        y.append(counts[i])
    # After last facility, extend flat to max budget
    if x[-1] < 175:
        x.append(175)
        y.append(y[-1])
    plt.plot(x, y, color=color, linewidth=2, label=label)
    plt.scatter(budgets, counts, color=color, s=10)

def create_step_plot(no_service_selection, full_service_selection):
    print("Creating true horizontal staircase plot...")
    plt.figure(figsize=(12, 8))

    if not no_service_selection.empty:
        create_true_staircase_plot(no_service_selection, 'b', 'No-Service Facilities')
    if not full_service_selection.empty:
        create_true_staircase_plot(full_service_selection, 'r', 'Full-Service Facilities')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Budget ($ Millions)', fontsize=14)
    plt.ylabel('Number of Facilities Selected', fontsize=14)
    plt.title('Continuous Budget Allocation: Number of Facilities vs. Budget', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.xlim(left=0, right=175)
    plt.ylim(bottom=0)
    output_folder = os.path.join(script_dir, "../results/figures")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "continuous_step_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    output_pdf = os.path.join(output_folder, "continuous_step_plot.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Continuous step plot saved to {output_file} and {output_pdf}")
    plt.close()

def save_complete_facility_tables(no_service_selection, full_service_selection):
    """
    Save complete tables of all facility selections with cumulative budget and coordinates
    """
    print("\nSaving complete facility selection tables...")
    
    # Create output folder
    output_folder = os.path.join(script_dir, "../results/continuous_selection")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save No-Service facility selections
    if not no_service_selection.empty:
        no_service_table = no_service_selection.copy()
        
        # Drop geometry column if it exists
        if 'geometry' in no_service_table.columns:
            no_service_table = no_service_table.drop(columns=['geometry'])
        
        no_service_table['Cost_Millions'] = no_service_table['Cost'] / 1e6
        no_service_table['Cumulative_Budget_Millions'] = no_service_table['Cumulative_Budget'] / 1e6
        
        # Create a clean table with relevant columns including coordinates
        no_service_clean = no_service_table[['Selection_Order', 'FID', 'Facility_Name', 'Cost_Millions', 
                                           'Cumulative_Budget_Millions', 'Capacity', 'Cumulative_Capacity',
                                           'centroid_x', 'centroid_y', 'Composite_Score']]
        
        # Save to CSV
        no_service_file = os.path.join(output_folder, "no_service_continuous_selection.csv")
        no_service_clean.to_csv(no_service_file, index=False, float_format='%.6f')
        
        print(f"No-Service facility table saved with {len(no_service_clean)} facilities")
        print(f"  Total budget used: ${no_service_clean['Cumulative_Budget_Millions'].max():.3f}M")
        print(f"  Total capacity provided: {no_service_clean['Cumulative_Capacity'].max()}")
    
    # Save Full-Service facility selections
    if not full_service_selection.empty:
        full_service_table = full_service_selection.copy()
        
        # Drop geometry column if it exists
        if 'geometry' in full_service_table.columns:
            full_service_table = full_service_table.drop(columns=['geometry'])
        
        full_service_table['Cost_Millions'] = full_service_table['Cost'] / 1e6
        full_service_table['Cumulative_Budget_Millions'] = full_service_table['Cumulative_Budget'] / 1e6
        
        # Create a clean table with relevant columns including coordinates
        full_service_clean = full_service_table[['Selection_Order', 'FID', 'Facility_Name', 'Cost_Millions', 
                                               'Cumulative_Budget_Millions', 'Capacity', 'Cumulative_Capacity',
                                               'centroid_x', 'centroid_y', 'Composite_Score']]
        
        # Save to CSV
        full_service_file = os.path.join(output_folder, "full_service_continuous_selection.csv")
        full_service_clean.to_csv(full_service_file, index=False, float_format='%.6f')
        
        print(f"Full-Service facility table saved with {len(full_service_clean)} facilities")
        print(f"  Total budget used: ${full_service_clean['Cumulative_Budget_Millions'].max():.3f}M")
        print(f"  Total capacity provided: {full_service_clean['Cumulative_Capacity'].max()}")
    
    print(f"Complete facility selection tables saved to {output_folder}")
    
    # Also save combined summary table
    create_combined_summary_table(no_service_selection, full_service_selection, output_folder)

def create_combined_summary_table(no_service_selection, full_service_selection, output_folder):
    """
    Create a combined summary table showing selection statistics for both scenarios
    """
    print("\nCreating combined summary table...")
    
    # Define budget points for summary
    budget_points = [2e6, 5e6, 10e6, 25e6, 50e6, 75e6, 100e6, 125e6, 150e6, 175e6]
    
    # Initialize summary data
    summary_data = []
    
    for budget in budget_points:
        budget_millions = budget / 1e6
        
        # Get no-service stats at this budget
        no_service_at_budget = no_service_selection[no_service_selection['Cumulative_Budget'] <= budget]
        no_service_count = len(no_service_at_budget)
        if no_service_count > 0:
            no_service_capacity = no_service_at_budget['Cumulative_Capacity'].max()
            no_service_used = no_service_at_budget['Cumulative_Budget'].max()
            no_service_remaining = budget - no_service_used
        else:
            no_service_capacity = 0
            no_service_used = 0
            no_service_remaining = budget
        
        # Get full-service stats at this budget
        full_service_at_budget = full_service_selection[full_service_selection['Cumulative_Budget'] <= budget]
        full_service_count = len(full_service_at_budget)
        if full_service_count > 0:
            full_service_capacity = full_service_at_budget['Cumulative_Capacity'].max()
            full_service_used = full_service_at_budget['Cumulative_Budget'].max()
            full_service_remaining = budget - full_service_used
        else:
            full_service_capacity = 0
            full_service_used = 0
            full_service_remaining = budget
        
        # Add to summary data
        summary_data.append({
            'Budget_Millions': budget_millions,
            'No_Service_Count': no_service_count,
            'No_Service_Capacity': no_service_capacity,
            'No_Service_Budget_Used_Millions': no_service_used / 1e6,
            'No_Service_Budget_Remaining_Millions': no_service_remaining / 1e6,
            'Full_Service_Count': full_service_count,
            'Full_Service_Capacity': full_service_capacity,
            'Full_Service_Budget_Used_Millions': full_service_used / 1e6,
            'Full_Service_Budget_Remaining_Millions': full_service_remaining / 1e6
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_file = os.path.join(output_folder, "budget_summary_statistics.csv")
    summary_df.to_csv(summary_file, index=False, float_format='%.6f')
    
    print(f"Combined summary table saved to {summary_file}")
    
def create_appendix_tables(no_service_selection, full_service_selection):
    """
    Create appendix tables showing which facilities were selected at each budget level,
    with detailed county information from candidates_new.shp
    """
    print("\nCreating appendix tables for facility selection...")
    
    # Define standard budget points
    budget_points = [2e6, 5e6, 10e6, 25e6, 50e6, 75e6, 100e6, 125e6, 150e6, 175e6]
    budget_columns = [f"${int(b/1e6)}M" for b in budget_points]
    
    # Create output folder
    output_folder = os.path.join(script_dir, "../results/appendices")
    os.makedirs(output_folder, exist_ok=True)
    
    # Load candidates_new.shp to get ComplexCom and CNTY_NM values
    try:
        candidates_path = "/Users/komalgulati/Documents/Project_3_1/simulation/datasets/candidates_new.shp"
        candidates_new_gdf = gpd.read_file(candidates_path)
        print(f"Loaded candidates_new.shp with {len(candidates_new_gdf)} features")
        
        # Print column names to verify CNTY_NM column exists
        print("Columns in candidates_new.shp:", candidates_new_gdf.columns.tolist())
        
        # Check if CNTY_NM exists, if not try alternative county name columns
        county_column = None
        possible_county_columns = ['CNTY_NM', 'COUNTY', 'COUNTY_NAME', 'CTY_NAME', 'CNTY_NAME', 'CO_NAME']
        
        for col in possible_county_columns:
            if col in candidates_new_gdf.columns:
                county_column = col
                print(f"Found county column: {col}")
                # Show sample values
                print(f"Sample county values from {col}:")
                print(candidates_new_gdf[col].head())
                break
        
        if county_column is None:
            print("WARNING: No county column found in candidates_new.shp!")
            # Look for any string columns that might contain county names
            string_cols = [col for col in candidates_new_gdf.columns if 
                         candidates_new_gdf[col].dtype == 'object' and 
                         col not in ['FID', 'ComplexNam', 'ComplexCom']]
            if string_cols:
                print(f"Potential string columns that might contain county info: {string_cols}")
                # Use the first one as fallback
                county_column = string_cols[0]
                print(f"Using {county_column} as fallback county column")
            else:
                print("No suitable county column found. County values will be empty.")
        
        # Ensure we have the necessary columns for joining
        if 'FID' not in candidates_new_gdf.columns:
            # Try alternative ID columns
            potential_id_fields = ['OBJECTID', 'ID', 'GID', 'OID']
            for field in potential_id_fields:
                if field in candidates_new_gdf.columns:
                    candidates_new_gdf['FID'] = candidates_new_gdf[field]
                    break
            else:
                # If no ID column found, use index
                candidates_new_gdf['FID'] = candidates_new_gdf.index
        
        # Convert FID to string for consistent joining
        candidates_new_gdf['FID'] = candidates_new_gdf['FID'].astype(str)
    except Exception as e:
        print(f"Error loading candidates_new.shp: {e}")
        return
    
    def create_appendix_for_scenario(selection_df, scenario_name):
        # Convert selection FID to string for consistent joining
        selection_df['FID'] = selection_df['FID'].astype(str)
        
        # Get unique list of all selected facility IDs and names
        # Remove duplicates by facility name (Facility_Name)
        unique_facilities = selection_df[['FID', 'Facility_Name']].drop_duplicates(subset=['Facility_Name'])

        
        # Merge with candidates_new_gdf to get both ComplexCom and county information
        if county_column:
            # Use both ComplexCom and county column
            appendix_df = unique_facilities.merge(
                candidates_new_gdf[['FID', 'ComplexCom', county_column]], 
                on='FID', 
                how='left'
            )
            
            # Rename columns to match expected format
            appendix_df = appendix_df.rename(columns={
                "Facility_Name": "ComplexNam", 
                "ComplexCom": "Interstate",
                county_column: "County"
            })
        else:
            # Just use ComplexCom if no county column found
            appendix_df = unique_facilities.merge(
                candidates_new_gdf[['FID', 'ComplexCom']], 
                on='FID', 
                how='left'
            )
            
            # Rename columns to match expected format
            appendix_df = appendix_df.rename(columns={
                "Facility_Name": "ComplexNam", 
                "ComplexCom": "Interstate"
            })
            # Add empty County column
            appendix_df['County'] = ''
        
        # Fill NaN with empty string
        appendix_df['County'] = appendix_df['County'].fillna('')
        appendix_df['Interstate'] = appendix_df['Interstate'].fillna('')
        # *** KEY LINE: REMOVE DUPLICATES BY COMPLEXNAM after the merge! ***
        appendix_df = appendix_df.drop_duplicates(subset=['ComplexNam'])
        # Debug information
        print(f"\nAppendix Summary for {scenario_name}:")
        print("Total facilities:", len(appendix_df))
        print("Facilities with Interstate values:", len(appendix_df[appendix_df['Interstate'] != '']))
        print("Facilities with County values:", len(appendix_df[appendix_df['County'] != '']))
        
        # Add budget columns (empty initially)
        for col in budget_columns:
            appendix_df[col] = ""
        
        # Mark which facilities were selected at each budget level
        for budget, col_name in zip(budget_points, budget_columns):
            # Get facilities selected at this budget
            selected_at_budget = selection_df[selection_df['Cumulative_Budget'] <= budget]
            
            if not selected_at_budget.empty:
                # Mark selected facilities with 'X'
                selected_fids = selected_at_budget['FID'].tolist()
                appendix_df.loc[appendix_df['FID'].isin(selected_fids), col_name] = "X"
        
        # Rearrange columns to match expected order
        column_order = ['ComplexNam', 'Interstate', 'County'] + budget_columns
        appendix_df = appendix_df[column_order]
        
        # Save to CSV
        appendix_file = os.path.join(output_folder, f"Appendix_{scenario_name.upper()}_Selected_Facilities.csv")
        appendix_df.to_csv(appendix_file, index=False)
        print(f"{scenario_name.capitalize()} scenario appendix saved to {appendix_file}")
        
        return appendix_df
    
    # Process both no-service and full-service scenarios
    if not no_service_selection.empty:
        create_appendix_for_scenario(no_service_selection, 'no_service')
    
    if not full_service_selection.empty:
        create_appendix_for_scenario(full_service_selection, 'full_service')
    
    print("Appendix tables created successfully.")
    return

def main():
    """
    Main function to run pure continuous budget allocation analysis up to $175M
    """
    print("Starting pure continuous budget allocation analysis...")
    
    # Load data
    candidates_gdf, existing_facilities = load_data()
    
    print(f"\n=== RUNNING CONTINUOUS BUDGET ALLOCATION UP TO ${MAX_BUDGET/1e6:.1f}M ===")
    
    # No-service scenario with continuous budget up to MAX_BUDGET
    no_service_selection = infinite_budget_allocation(
        candidates_gdf, 
        existing_facilities, 
        'cost_no_service', 
        'no_service',
        MAX_BUDGET
    )
    
    # Full-service scenario with continuous budget up to MAX_BUDGET
    full_service_selection = infinite_budget_allocation(
        candidates_gdf, 
        existing_facilities, 
        'cost_full_service', 
        'full_service',
        MAX_BUDGET
    )
    
    # Create continuous step plot showing every facility addition
    create_step_plot(no_service_selection, full_service_selection)
    
    # Save complete facility selection tables
    save_complete_facility_tables(no_service_selection, full_service_selection)
    
    # Create appendix tables showing facilities selected at each budget level
    create_appendix_tables(no_service_selection, full_service_selection)
    
    print("\nPure continuous budget allocation analysis completed!")

if __name__ == "__main__":
    main()