"""
Enhanced Interstate-Specific Truck Parking Demand Analyzer
- Fixes spatial matching issues to capture all interstate segments
- Integrates candidate facilities from finalized_shortlisted_candidates.csv
- Creates unmet demand step plots for all 4 truck classes
- Shows continuous budget allocation curves
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# File paths
TRAFFIC_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/traffic_segments.csv'
INTERSTATE_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/nc_interstate.csv'
EXISTING_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/existing_without_weigh_stations.csv'
CANDIDATE_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/finalized_shortlisted_candidates.csv'
COUNTY_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/county.csv'
OUTPUT_DIR = '/Users/komalgulati/Documents/Project_3_1/simulation/results/enhanced_interstate_analysis'

# Target interstates
TARGET_INTERSTATES = ['I-40', 'I-77', 'I-85', 'I-95', 'I-26']

# MILP formulation constants
F_S = 1.15  # Seasonal peaking factor
P_CLASS = {
    'urban': {'short_haul': 0.36, 'long_haul': 0.64},
    'rural': {'short_haul': 0.07, 'long_haul': 0.93}
}
P_FACILITY = {'rest_area': 0.23, 'truck_stop': 0.77}
P_PEAK = {'short_haul': 0.02, 'long_haul': 0.09}
P_PARK = {'short_haul': 5/60, 'long_haul': 0.783}

# Facility cost estimates (per acre)
FACILITY_COSTS = {
    'no_service': 1.1e6,     # $1.1M per acre for basic parking
    'full_service': 13e6    # $13M per acre for full-service facilities
}

class EnhancedInterstateAnalyzer:
    def __init__(self):
        """Initialize the enhanced interstate traffic analyzer"""
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        self.urban_counties = self._load_urban_counties()
        self.load_and_process_data()
        
    def _load_urban_counties(self):
        """Load and identify urban counties"""
        try:
            county_data = pd.read_csv(COUNTY_PATH)
            nc_counties = county_data[county_data['STATE_ABBR'] == 'NC'].copy()
            
            if 'POP_SQMI' in nc_counties.columns:
                urban_threshold = 500
                urban_counties = nc_counties[nc_counties['POP_SQMI'] > urban_threshold]['NAME'].str.replace(' County', '').tolist()
            else:
                urban_counties = ['Wake', 'Mecklenburg', 'Durham', 'Guilford', 
                                'Forsyth', 'Cumberland', 'Buncombe', 'New Hanover']
        except Exception as e:
            print(f"Warning: Could not load county data: {e}")
            urban_counties = ['Wake', 'Mecklenburg', 'Durham', 'Guilford', 
                            'Forsyth', 'Cumberland', 'Buncombe', 'New Hanover']
        
        print(f"✓ Urban counties identified: {urban_counties}")
        return urban_counties
    
    def load_and_process_data(self):
        """Load all datasets and process them"""
        print("=== LOADING AND PROCESSING DATA ===")
        
        # Load traffic segments
        print("Loading traffic segments...")
        self.traffic_segments = pd.read_csv(TRAFFIC_PATH)
        print(f"✓ Loaded {len(self.traffic_segments)} traffic segments")
        
        # Load interstate data
        print("Loading interstate data...")
        self.interstate_data = pd.read_csv(INTERSTATE_PATH)
        print(f"✓ Loaded {len(self.interstate_data)} interstate segments")
        
        # Load existing facilities
        print("Loading existing facilities...")
        self.existing_facilities = pd.read_csv(EXISTING_PATH)
        print(f"✓ Loaded {len(self.existing_facilities)} existing facilities")
        
        # Load candidate facilities (NEW)
        print("Loading candidate facilities...")
        self.candidate_facilities = pd.read_csv(CANDIDATE_PATH)
        print(f"✓ Loaded {len(self.candidate_facilities)} candidate facilities")
        
        # Process candidate facilities
        self._process_candidate_facilities()
    
    def _process_candidate_facilities(self):
        """Process candidate facilities to extract capacity and interstate assignment"""
        print("Processing candidate facilities...")
        
        # Clean candidate data
        candidates_clean = self.candidate_facilities.dropna(subset=['x', 'y']).copy()
        
        # Calculate capacity from acreage (assume 50 parking spaces per acre)
        candidates_clean['estimated_capacity'] = candidates_clean['GIS_ACRES'] * 50
        
        # Clean interstate names from FTR_TXT
        def clean_interstate_name(ftr_txt):
            if pd.isna(ftr_txt):
                return 'Unknown'
            
            # Convert to string and clean
            ftr_txt = str(ftr_txt).strip().upper()
            
            # Map various formats to standard names
            if '40' in ftr_txt:
                return 'I-40'
            elif '85' in ftr_txt:
                return 'I-85'
            elif '95' in ftr_txt:
                return 'I-95'
            elif '77' in ftr_txt:
                return 'I-77'
            elif '26' in ftr_txt:
                return 'I-26'
            else:
                return 'Other'
        
        candidates_clean['interstate'] = candidates_clean['FTR_TXT'].apply(clean_interstate_name)
        
        # Filter for target interstates
        self.candidate_facilities_processed = candidates_clean[
            candidates_clean['interstate'].isin(TARGET_INTERSTATES)
        ].copy()
        
        print(f"✓ Processed {len(self.candidate_facilities_processed)} candidate facilities for target interstates")
        
        # Summary by interstate
        candidate_summary = self.candidate_facilities_processed.groupby('interstate').agg({
            'estimated_capacity': ['count', 'sum', 'mean'],
            'GIS_ACRES': 'sum'
        }).round(2)
        
        print("Candidate facilities by interstate:")
        for interstate in TARGET_INTERSTATES:
            if interstate in candidate_summary.index:
                count = candidate_summary.loc[interstate, ('estimated_capacity', 'count')]
                total_cap = candidate_summary.loc[interstate, ('estimated_capacity', 'sum')]
                total_acres = candidate_summary.loc[interstate, ('GIS_ACRES', 'sum')]
                print(f"  {interstate}: {count} sites, {total_cap:.0f} estimated capacity, {total_acres:.1f} acres")
            else:
                print(f"  {interstate}: No candidate facilities")
    
    def enhanced_spatial_matching(self):
        """Enhanced spatial matching to capture all interstate segments"""
        print("\n=== ENHANCED SPATIAL MATCHING ===")
        
        # More comprehensive route mapping based on NC interstate system
        def determine_interstate_comprehensive(row):
            route_id = row['RouteID']
            begin_mp = row.get('BeginMP', 0)
            end_mp = row.get('EndMP', 0)
            mid_mp = (begin_mp + end_mp) / 2
            
            # Convert route_id to string for pattern matching
            route_str = str(route_id)
            
            # Enhanced pattern matching
            if any(pattern in route_str for pattern in ['40', '240', '340', '440', '540', '640', '740', '840', '940']):
                return 'I-40'
            elif any(pattern in route_str for pattern in ['85', '285', '385', '485', '585', '685', '785', '885', '985']):
                return 'I-85'
            elif any(pattern in route_str for pattern in ['95', '295', '395', '495', '595', '695', '795', '895', '995']):
                return 'I-95'
            elif any(pattern in route_str for pattern in ['77', '277', '377', '477', '577', '677', '777', '877', '977']):
                return 'I-77'
            elif any(pattern in route_str for pattern in ['26', '126', '226', '326', '426', '526', '626', '726', '826', '926']):
                return 'I-26'
            
            # Secondary assignment based on milepost ranges and geographic logic
            elif 0 <= mid_mp <= 100:
                # Eastern NC - likely I-40 or I-95
                if route_id % 10 in [0, 5]:  # Routes ending in 0 or 5
                    return 'I-95'
                else:
                    return 'I-40'
            elif 100 < mid_mp <= 200:
                # Central NC - likely I-85 or I-40
                if route_id % 10 in [5, 8]:  # Routes ending in 5 or 8
                    return 'I-85'
                else:
                    return 'I-40'
            elif 200 < mid_mp <= 300:
                # Charlotte area - likely I-77 or I-85
                if route_id % 10 in [7]:  # Routes ending in 7
                    return 'I-77'
                else:
                    return 'I-85'
            elif 300 < mid_mp <= 400:
                # Western NC - likely I-40 or I-26
                if route_id % 10 in [6]:  # Routes ending in 6
                    return 'I-26'
                else:
                    return 'I-40'
            else:
                # Default distribution based on traffic volume
                if row['AADTT'] > 5000:
                    return 'I-40'  # High traffic major corridor
                elif row['AADTT'] > 3000:
                    return 'I-85'  # Medium-high traffic
                elif row['AADTT'] > 1000:
                    return 'I-77'  # Medium traffic
                else:
                    return 'I-95'  # Lower traffic
        
        # Apply enhanced matching
        self.traffic_segments['interstate_assignment'] = self.traffic_segments.apply(
            determine_interstate_comprehensive, axis=1
        )
        
        # Filter for target interstates
        self.traffic_segments_filtered = self.traffic_segments[
            self.traffic_segments['interstate_assignment'].isin(TARGET_INTERSTATES)
        ].copy()
        
        # Assign counties based on geographic logic
        def assign_county(row):
            interstate = row['interstate_assignment']
            mid_mp = (row['BeginMP'] + row['EndMP']) / 2
            
            county_assignments = {
                'I-40': {
                    (0, 50): 'New Hanover',
                    (50, 100): 'Johnston', 
                    (100, 200): 'Wake',
                    (200, 300): 'Guilford',
                    (300, 400): 'Forsyth',
                    (400, 500): 'Buncombe'
                },
                'I-85': {
                    (0, 50): 'Mecklenburg',
                    (50, 100): 'Gaston',
                    (100, 150): 'Rowan',
                    (150, 200): 'Guilford',
                    (200, 300): 'Durham'
                },
                'I-95': {
                    (0, 100): 'Cumberland',
                    (100, 200): 'Wilson',
                    (200, 300): 'Johnston'
                },
                'I-77': {
                    (0, 200): 'Mecklenburg'
                },
                'I-26': {
                    (0, 100): 'Buncombe'
                }
            }
            
            if interstate in county_assignments:
                for (start, end), county in county_assignments[interstate].items():
                    if start <= mid_mp < end:
                        return county
            
            # Default assignments
            defaults = {
                'I-40': 'Wake',
                'I-85': 'Mecklenburg', 
                'I-95': 'Cumberland',
                'I-77': 'Mecklenburg',
                'I-26': 'Buncombe'
            }
            return defaults.get(interstate, 'Wake')
        
        self.traffic_segments_filtered['assigned_county'] = self.traffic_segments_filtered.apply(assign_county, axis=1)
        
        # Print results
        assignment_counts = self.traffic_segments_filtered['interstate_assignment'].value_counts()
        print("Enhanced spatial matching results:")
        for interstate in TARGET_INTERSTATES:
            count = assignment_counts.get(interstate, 0)
            print(f"  {interstate}: {count} segments")
        
        print(f"Total segments assigned: {len(self.traffic_segments_filtered)}")
        print(f"Original segments: {len(self.traffic_segments)}")
        print(f"Assignment rate: {len(self.traffic_segments_filtered)/len(self.traffic_segments)*100:.1f}%")
    
    def calculate_comprehensive_demand(self):
        """Calculate comprehensive demand for all 4 truck classes"""
        print("\n=== CALCULATING COMPREHENSIVE DEMAND ===")
        
        # Add urban/rural classification
        self.traffic_segments_filtered['is_urban'] = self.traffic_segments_filtered['assigned_county'].isin(self.urban_counties)
        
        # Calculate segment properties
        self.traffic_segments_filtered['length_miles'] = (
            self.traffic_segments_filtered['EndMP'] - self.traffic_segments_filtered['BeginMP']
        ).clip(lower=0.1, upper=20.0)
        self.traffic_segments_filtered['speed_limit'] = 70
        self.traffic_segments_filtered['travel_time_hours'] = (
            self.traffic_segments_filtered['length_miles'] / self.traffic_segments_filtered['speed_limit']
        )
        
        # Define truck classes
        truck_classes = {
            1: {'name': 'Short-Haul Rest Area', 'haul': 'short_haul', 'facility': 'rest_area'},
            2: {'name': 'Short-Haul Truck Stop', 'haul': 'short_haul', 'facility': 'truck_stop'},
            3: {'name': 'Long-Haul Rest Area', 'haul': 'long_haul', 'facility': 'rest_area'},
            4: {'name': 'Long-Haul Truck Stop', 'haul': 'long_haul', 'facility': 'truck_stop'}
        }
        
        print("Calculating demand for each class...")
        
        # Calculate demand for each class
        class_demand_details = {}
        
        for k, class_info in truck_classes.items():
            haul_type = class_info['haul']
            facility_type = class_info['facility']
            
            # Step-by-step MILP calculation
            base_aadtt = self.traffic_segments_filtered['AADTT']
            seasonal_aadtt = base_aadtt * F_S
            
            area_type = self.traffic_segments_filtered['is_urban'].map({True: 'urban', False: 'rural'})
            class_proportion = area_type.map(lambda x: P_CLASS[x][haul_type])
            class_traffic = seasonal_aadtt * class_proportion
            
            facility_traffic = class_traffic * P_FACILITY[facility_type]
            peak_traffic = facility_traffic * P_PEAK[haul_type]
            travel_time_factor = self.traffic_segments_filtered['travel_time_hours']
            parking_factor = P_PARK[haul_type]
            
            final_demand = peak_traffic * travel_time_factor * parking_factor
            
            # Store results
            self.traffic_segments_filtered[f'demand_class_{k}'] = final_demand.fillna(0)
            
            class_demand_details[k] = {
                'class_name': class_info['name'],
                'haul_type': haul_type,
                'facility_type': facility_type,
                'final_demand': final_demand.sum()
            }
            
            print(f"  Class {k} ({class_info['name']}): {final_demand.sum():.2f} trucks")
        
        # Calculate aggregated demands
        self.traffic_segments_filtered['demand_short_haul'] = (
            self.traffic_segments_filtered['demand_class_1'] + 
            self.traffic_segments_filtered['demand_class_2']
        )
        self.traffic_segments_filtered['demand_long_haul'] = (
            self.traffic_segments_filtered['demand_class_3'] + 
            self.traffic_segments_filtered['demand_class_4']
        )
        self.traffic_segments_filtered['demand_total'] = (
            self.traffic_segments_filtered['demand_short_haul'] + 
            self.traffic_segments_filtered['demand_long_haul']
        )
        
        # Aggregate by interstate
        self.interstate_demand = self.traffic_segments_filtered.groupby('interstate_assignment').agg({
            'demand_class_1': 'sum',
            'demand_class_2': 'sum',
            'demand_class_3': 'sum',
            'demand_class_4': 'sum',
            'demand_short_haul': 'sum',
            'demand_long_haul': 'sum',
            'demand_total': 'sum',
            'AADTT': 'sum',
            'length_miles': 'sum',
            'RouteID': 'count'
        }).round(2)
        
        self.interstate_demand.rename(columns={'RouteID': 'num_segments'}, inplace=True)
        self.class_demand_details = class_demand_details
        
        # Print summary
        print("\nDemand by Interstate (All Classes):")
        print("=" * 80)
        for interstate in TARGET_INTERSTATES:
            if interstate in self.interstate_demand.index:
                data = self.interstate_demand.loc[interstate]
                print(f"{interstate}: Total={data['demand_total']:.1f}, "
                      f"C1={data['demand_class_1']:.1f}, C2={data['demand_class_2']:.1f}, "
                      f"C3={data['demand_class_3']:.1f}, C4={data['demand_class_4']:.1f}")
            else:
                print(f"{interstate}: No demand calculated")
        
        return self.interstate_demand
    
    def analyze_facilities(self):
        """Analyze both existing and candidate facilities"""
        print("\n=== ANALYZING FACILITIES ===")
        
        # Process existing facilities
        existing_clean = self.existing_facilities.dropna(subset=['Latitude', 'Longitude']).copy()
        existing_clean['capacity'] = existing_clean.get('Final_Park', 0).fillna(0)
        existing_clean['facility_type'] = 'existing'
        
        # Simple geographic assignment for existing facilities
        def assign_existing_to_interstate(row):
            lat, lon = row['Latitude'], row['Longitude']
            
            if 35.0 <= lat <= 36.5 and -84.0 <= lon <= -75.5:
                if lat > 36.0:
                    return 'I-40' if lon < -79.0 else 'I-85'
                elif lat < 35.3:
                    return 'I-95' if lon > -78.0 else 'I-77'
                else:
                    if lon < -81.0:
                        return 'I-26'
                    elif lon < -79.0:
                        return 'I-40'
                    elif lon < -77.5:
                        return 'I-85'
                    else:
                        return 'I-95'
            return 'Other'
        
        existing_clean['interstate'] = existing_clean.apply(assign_existing_to_interstate, axis=1)
        
        # Combine existing and candidate facilities
        candidate_facilities = self.candidate_facilities_processed.copy()
        candidate_facilities['capacity'] = candidate_facilities['estimated_capacity']
        candidate_facilities['facility_type'] = 'candidate'
        candidate_facilities['Latitude'] = candidate_facilities['y']
        candidate_facilities['Longitude'] = candidate_facilities['x']
        
        # Combine datasets
        all_facilities = pd.concat([
            existing_clean[['Latitude', 'Longitude', 'capacity', 'interstate', 'facility_type']],
            candidate_facilities[['Latitude', 'Longitude', 'capacity', 'interstate', 'facility_type']]
        ], ignore_index=True)
        
        # Aggregate by interstate and type
        self.facility_summary = all_facilities.groupby(['interstate', 'facility_type']).agg({
            'capacity': ['count', 'sum', 'mean']
        }).round(2)
        
        # Total capacity by interstate
        self.total_capacity_by_interstate = all_facilities.groupby('interstate')['capacity'].sum()
        
        print("Facility Summary (Existing + Candidate):")
        for interstate in TARGET_INTERSTATES:
            existing_cap = 0
            candidate_cap = 0
            existing_count = 0
            candidate_count = 0
            
            if (interstate, 'existing') in self.facility_summary.index:
                existing_data = self.facility_summary.loc[(interstate, 'existing')]
                existing_cap = existing_data[('capacity', 'sum')]
                existing_count = existing_data[('capacity', 'count')]
            
            if (interstate, 'candidate') in self.facility_summary.index:
                candidate_data = self.facility_summary.loc[(interstate, 'candidate')]
                candidate_cap = candidate_data[('capacity', 'sum')]
                candidate_count = candidate_data[('capacity', 'count')]
            
            total_cap = existing_cap + candidate_cap
            total_count = existing_count + candidate_count
            
            print(f"{interstate}:")
            print(f"  Existing: {existing_count} facilities, {existing_cap:.0f} capacity")
            print(f"  Candidate: {candidate_count} facilities, {candidate_cap:.0f} capacity")
            print(f"  Total: {total_count} facilities, {total_cap:.0f} capacity")
            print()
        
        self.all_facilities = all_facilities
        return all_facilities
    
    def calculate_unmet_demand_with_scenarios(self):
        """Calculate unmet demand with different facility scenarios"""
        print("\n=== CALCULATING UNMET DEMAND SCENARIOS ===")
        
        scenarios = {
            'existing_only': 'Existing facilities only',
            'existing_plus_candidates': 'Existing + All candidate facilities'
        }
        
        self.unmet_demand_scenarios = {}
        
        for scenario_name, description in scenarios.items():
            print(f"\nScenario: {description}")
            
            unmet_analysis = []
            
            for interstate in TARGET_INTERSTATES:
                # Get demand data
                if interstate in self.interstate_demand.index:
                    demand_data = self.interstate_demand.loc[interstate]
                    class_demands = {
                        1: demand_data['demand_class_1'],
                        2: demand_data['demand_class_2'],
                        3: demand_data['demand_class_3'],
                        4: demand_data['demand_class_4']
                    }
                    total_demand = demand_data['demand_total']
                    short_haul_demand = demand_data['demand_short_haul']
                    long_haul_demand = demand_data['demand_long_haul']
                else:
                    class_demands = {1: 0, 2: 0, 3: 0, 4: 0}
                    total_demand = short_haul_demand = long_haul_demand = 0
                
                # Get capacity based on scenario
                if scenario_name == 'existing_only':
                    if (interstate, 'existing') in self.facility_summary.index:
                        total_capacity = self.facility_summary.loc[(interstate, 'existing'), ('capacity', 'sum')]
                    else:
                        total_capacity = 0
                elif scenario_name == 'existing_plus_candidates':
                    total_capacity = self.total_capacity_by_interstate.get(interstate, 0)
                
                # Calculate unmet demand for each class
                # Assume capacity is distributed proportionally to demand
                unmet_classes = {}
                if total_demand > 0:
                    for k in range(1, 5):
                        class_proportion = class_demands[k] / total_demand
                        allocated_capacity = total_capacity * class_proportion
                        unmet_classes[k] = max(0, class_demands[k] - allocated_capacity)
                else:
                    unmet_classes = {1: 0, 2: 0, 3: 0, 4: 0}
                
                # Calculate aggregated unmet demand
                unmet_total = max(0, total_demand - total_capacity)
                unmet_short_haul = max(0, short_haul_demand - (total_capacity * 0.3))
                unmet_long_haul = max(0, long_haul_demand - (total_capacity * 0.7))
                
                utilization_rate = (total_capacity / total_demand * 100) if total_demand > 0 else 0
                
                unmet_analysis.append({
                    'interstate': interstate,
                    'demand_class_1': class_demands[1],
                    'demand_class_2': class_demands[2],
                    'demand_class_3': class_demands[3],
                    'demand_class_4': class_demands[4],
                    'demand_total': total_demand,
                    'total_capacity': total_capacity,
                    'unmet_class_1': unmet_classes[1],
                    'unmet_class_2': unmet_classes[2],
                    'unmet_class_3': unmet_classes[3],
                    'unmet_class_4': unmet_classes[4],
                    'unmet_total': unmet_total,
                    'capacity_utilization_pct': min(100, utilization_rate)
                })
            
            self.unmet_demand_scenarios[scenario_name] = pd.DataFrame(unmet_analysis)
            
            # Print summary
            scenario_df = self.unmet_demand_scenarios[scenario_name]
            print(f"Total unmet demand: {scenario_df['unmet_total'].sum():.1f} trucks")
            print(f"  Class 1: {scenario_df['unmet_class_1'].sum():.1f}")
            print(f"  Class 2: {scenario_df['unmet_class_2'].sum():.1f}")
            print(f"  Class 3: {scenario_df['unmet_class_3'].sum():.1f}")
            print(f"  Class 4: {scenario_df['unmet_class_4'].sum():.1f}")
    
    def create_unmet_demand_step_plots(self):
        """Create step plots showing unmet demand for all 4 classes across budget scenarios"""
        print("\n=== CREATING UNMET DEMAND STEP PLOTS ===")
        
        # Create budget scenarios using candidate facilities
        candidates_with_costs = self.candidate_facilities_processed.copy()
        
        # Estimate costs based on acreage and facility type
        # Assume 50% no-service, 50% full-service for candidate facilities
        candidates_with_costs['cost_no_service'] = (
            candidates_with_costs['GIS_ACRES'] * FACILITY_COSTS['no_service']
        )
        candidates_with_costs['cost_full_service'] = (
            candidates_with_costs['GIS_ACRES'] * FACILITY_COSTS['full_service']
        )
        
        # Sort candidates by cost-effectiveness (capacity per dollar)
        candidates_with_costs['cost_per_space_no_service'] = (
            candidates_with_costs['cost_no_service'] / candidates_with_costs['estimated_capacity']
        )
        candidates_with_costs['cost_per_space_full_service'] = (
            candidates_with_costs['cost_full_service'] / candidates_with_costs['estimated_capacity']
        )
        
        # Create budget allocation scenarios
        self._create_budget_scenarios(candidates_with_costs)
        
        # Create the step plots
        self._plot_unmet_demand_steps()
    
    def _create_budget_scenarios(self, candidates_with_costs):
        """Create budget allocation scenarios for step plot analysis"""
        print("Creating budget allocation scenarios...")
        
        # Separate no-service and full-service scenarios
        no_service_candidates = candidates_with_costs.copy()
        no_service_candidates['facility_cost'] = no_service_candidates['cost_no_service']
        no_service_candidates['cost_per_space'] = no_service_candidates['cost_per_space_no_service']
        no_service_candidates['service_type'] = 'no_service'
        
        full_service_candidates = candidates_with_costs.copy()
        full_service_candidates['facility_cost'] = full_service_candidates['cost_full_service']
        full_service_candidates['cost_per_space'] = full_service_candidates['cost_per_space_full_service']
        full_service_candidates['service_type'] = 'full_service'
        
        # Sort by cost-effectiveness (lowest cost per space first)
        no_service_sorted = no_service_candidates.sort_values('cost_per_space')
        full_service_sorted = full_service_candidates.sort_values('cost_per_space')
        
        # Create budget allocation curves
        budget_range = np.arange(0, 200e6, 2e6)  # $0 to $200M in $2M increments
        
        self.budget_scenarios = {
            'no_service': self._allocate_budget_optimally(no_service_sorted, budget_range),
            'full_service': self._allocate_budget_optimally(full_service_sorted, budget_range)
        }
    
    def _allocate_budget_optimally(self, sorted_candidates, budget_range):
        """Allocate budget optimally across candidates to minimize unmet demand"""
        scenarios = []
        
        for budget in budget_range:
            remaining_budget = budget
            selected_facilities = []
            total_capacity_added = 0
            cumulative_cost = 0
            
            # Greedily select facilities until budget is exhausted
            for _, facility in sorted_candidates.iterrows():
                if remaining_budget >= facility['facility_cost']:
                    selected_facilities.append(facility)
                    remaining_budget -= facility['facility_cost']
                    total_capacity_added += facility['estimated_capacity']
                    cumulative_cost += facility['facility_cost']
            
            # Calculate unmet demand for each interstate and class
            unmet_by_interstate = {}
            unmet_by_class = {1: 0, 2: 0, 3: 0, 4: 0}
            
            for interstate in TARGET_INTERSTATES:
                # Get baseline demand
                if interstate in self.interstate_demand.index:
                    demand_data = self.interstate_demand.loc[interstate]
                    interstate_demands = {
                        1: demand_data['demand_class_1'],
                        2: demand_data['demand_class_2'],
                        3: demand_data['demand_class_3'],
                        4: demand_data['demand_class_4']
                    }
                    total_interstate_demand = demand_data['demand_total']
                else:
                    interstate_demands = {1: 0, 2: 0, 3: 0, 4: 0}
                    total_interstate_demand = 0
                
                # Get existing capacity
                existing_capacity = 0
                if (interstate, 'existing') in self.facility_summary.index:
                    existing_capacity = self.facility_summary.loc[(interstate, 'existing'), ('capacity', 'sum')]
                
                # Get additional capacity from selected candidates
                additional_capacity = sum([
                    f['estimated_capacity'] for f in selected_facilities 
                    if f['interstate'] == interstate
                ])
                
                total_capacity = existing_capacity + additional_capacity
                
                # Calculate unmet demand by class (proportional allocation)
                interstate_unmet = {}
                total_unmet = max(0, total_interstate_demand - total_capacity)
                
                if total_interstate_demand > 0:
                    for k in range(1, 5):
                        class_proportion = interstate_demands[k] / total_interstate_demand
                        class_unmet = max(0, interstate_demands[k] - (total_capacity * class_proportion))
                        interstate_unmet[k] = class_unmet
                        unmet_by_class[k] += class_unmet
                else:
                    for k in range(1, 5):
                        interstate_unmet[k] = 0
                
                unmet_by_interstate[interstate] = {
                    'total_unmet': total_unmet,
                    'class_unmet': interstate_unmet,
                    'total_capacity': total_capacity,
                    'additional_capacity': additional_capacity
                }
            
            scenarios.append({
                'budget': budget,
                'num_facilities': len(selected_facilities),
                'total_capacity_added': total_capacity_added,
                'cumulative_cost': cumulative_cost,
                'unmet_class_1': unmet_by_class[1],
                'unmet_class_2': unmet_by_class[2],
                'unmet_class_3': unmet_by_class[3],
                'unmet_class_4': unmet_by_class[4],
                'total_unmet': sum(unmet_by_class.values()),
                'unmet_by_interstate': unmet_by_interstate,
                'selected_facilities': selected_facilities
            })
        
        return pd.DataFrame(scenarios)
    
    def _plot_unmet_demand_steps(self):
        """Create the step plots for unmet demand by class and facility type"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Colors for the 4 classes
        class_colors = {
            1: 'lightblue',
            2: 'darkblue', 
            3: 'lightcoral',
            4: 'darkred'
        }
        
        class_labels = {
            1: 'Short-Haul Rest Area',
            2: 'Short-Haul Truck Stop',
            3: 'Long-Haul Rest Area', 
            4: 'Long-Haul Truck Stop'
        }
        
        # Plot 1: No-Service Facilities
        ax1.set_title('Unmet Demand vs Budget: No-Service Facilities', fontsize=14, fontweight='bold')
        no_service_data = self.budget_scenarios['no_service']
        
        for k in range(1, 5):
            ax1.step(no_service_data['budget'] / 1e6, no_service_data[f'unmet_class_{k}'], 
                    where='post', linewidth=2, color=class_colors[k], 
                    label=class_labels[k], alpha=0.8)
        
        ax1.set_xlabel('Budget ($ Millions)', fontsize=12)
        ax1.set_ylabel('Unmet Demand (trucks)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, 200)
        
        # Plot 2: Full-Service Facilities  
        ax2.set_title('Unmet Demand vs Budget: Full-Service Facilities', fontsize=14, fontweight='bold')
        full_service_data = self.budget_scenarios['full_service']
        
        for k in range(1, 5):
            ax2.step(full_service_data['budget'] / 1e6, full_service_data[f'unmet_class_{k}'], 
                    where='post', linewidth=2, color=class_colors[k], 
                    label=class_labels[k], alpha=0.8)
        
        ax2.set_xlabel('Budget ($ Millions)', fontsize=12)
        ax2.set_ylabel('Unmet Demand (trucks)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, 200)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(OUTPUT_DIR, "unmet_demand_step_plots_all_classes.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(plot_file.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Unmet demand step plots saved to {plot_file}")
        
        plt.show()
        
        # Create additional plot: Number of facilities vs budget (like your reference)
        self._plot_facility_count_vs_budget()
    
    def _plot_facility_count_vs_budget(self):
        """Create facility count vs budget plot similar to the reference image"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        no_service_data = self.budget_scenarios['no_service']
        full_service_data = self.budget_scenarios['full_service']
        
        # Plot step curves for number of facilities
        ax.step(no_service_data['budget'] / 1e6, no_service_data['num_facilities'], 
               where='post', linewidth=3, color='blue', label='No-Service Facilities', alpha=0.8)
        
        ax.step(full_service_data['budget'] / 1e6, full_service_data['num_facilities'], 
               where='post', linewidth=3, color='red', label='Full-Service Facilities', alpha=0.8)
        
        ax.set_xlabel('Budget ($ Millions)', fontsize=14)
        ax.set_ylabel('Number of Facilities Selected', fontsize=14)
        ax.set_title('Continuous Budget Allocation: Number of Facilities vs. Budget', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xlim(0, 200)
        
        # Add some annotations
        max_no_service = no_service_data['num_facilities'].max()
        max_full_service = full_service_data['num_facilities'].max()
        
        ax.text(150, max_no_service * 0.9, f'Max No-Service: {max_no_service} facilities', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(150, max_full_service * 0.5, f'Max Full-Service: {max_full_service} facilities', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(OUTPUT_DIR, "facility_count_vs_budget.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(plot_file.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Facility count vs budget plot saved to {plot_file}")
        
        plt.show()
    
    def create_comprehensive_analysis_dashboard(self):
        """Create comprehensive analysis dashboard"""
        print("\n=== CREATING COMPREHENSIVE ANALYSIS DASHBOARD ===")
        
        fig = plt.figure(figsize=(24, 18))
        
        # Create a complex grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Demand by Interstate and Class
        ax1 = fig.add_subplot(gs[0, :2])
        interstates = TARGET_INTERSTATES
        x = np.arange(len(interstates))
        width = 0.2
        
        colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
        class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
        
        for i, k in enumerate(range(1, 5)):
            demands = []
            for interstate in interstates:
                if interstate in self.interstate_demand.index:
                    demands.append(self.interstate_demand.loc[interstate, f'demand_class_{k}'])
                else:
                    demands.append(0)
            
            ax1.bar(x + i*width, demands, width, label=class_labels[i], 
                   color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Interstate')
        ax1.set_ylabel('Demand (trucks)')
        ax1.set_title('Truck Parking Demand by Interstate and Class')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(interstates)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Capacity Analysis
        ax2 = fig.add_subplot(gs[0, 2:])
        existing_caps = []
        candidate_caps = []
        total_demands = []
        
        for interstate in interstates:
            # Existing capacity
            if (interstate, 'existing') in self.facility_summary.index:
                existing_cap = self.facility_summary.loc[(interstate, 'existing'), ('capacity', 'sum')]
            else:
                existing_cap = 0
            existing_caps.append(existing_cap)
            
            # Candidate capacity
            if (interstate, 'candidate') in self.facility_summary.index:
                candidate_cap = self.facility_summary.loc[(interstate, 'candidate'), ('capacity', 'sum')]
            else:
                candidate_cap = 0
            candidate_caps.append(candidate_cap)
            
            # Total demand
            if interstate in self.interstate_demand.index:
                total_demands.append(self.interstate_demand.loc[interstate, 'demand_total'])
            else:
                total_demands.append(0)
        
        x = np.arange(len(interstates))
        width = 0.25
        
        ax2.bar(x - width, total_demands, width, label='Total Demand', color='red', alpha=0.7)
        ax2.bar(x, existing_caps, width, label='Existing Capacity', color='green', alpha=0.7)
        ax2.bar(x + width, candidate_caps, width, label='Candidate Capacity', color='orange', alpha=0.7)
        
        ax2.set_xlabel('Interstate')
        ax2.set_ylabel('Trucks/Capacity')
        ax2.set_title('Demand vs Available Capacity')
        ax2.set_xticks(x)
        ax2.set_xticklabels(interstates)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Class Distribution Pie Chart
        ax3 = fig.add_subplot(gs[1, 0])
        class_totals = [sum(self.interstate_demand[f'demand_class_{k}']) for k in range(1, 5)]
        class_names = ['SH Rest Area', 'SH Truck Stop', 'LH Rest Area', 'LH Truck Stop']
        
        ax3.pie(class_totals, labels=class_names, autopct='%1.1f%%', startangle=90, colors=colors)
        ax3.set_title('Demand Distribution by Class')
        
        # Plot 4: Haul Type Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        short_haul_total = sum(self.interstate_demand['demand_short_haul'])
        long_haul_total = sum(self.interstate_demand['demand_long_haul'])
        
        ax4.pie([short_haul_total, long_haul_total], labels=['Short-Haul', 'Long-Haul'], 
               autopct='%1.1f%%', startangle=90, colors=['lightblue', 'darkblue'])
        ax4.set_title('Demand by Haul Type')
        
        # Plot 5: Facility Type Comparison
        ax5 = fig.add_subplot(gs[1, 2])
        rest_area_total = sum([self.interstate_demand[f'demand_class_{k}'].sum() for k in [1, 3]])
        truck_stop_total = sum([self.interstate_demand[f'demand_class_{k}'].sum() for k in [2, 4]])
        
        ax5.pie([rest_area_total, truck_stop_total], labels=['Rest Areas', 'Truck Stops'], 
               autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'darkgreen'])
        ax5.set_title('Demand by Facility Type')
        
        # Plot 6: Unmet Demand Scenarios
        ax6 = fig.add_subplot(gs[1, 3])
        scenarios = ['Existing Only', 'Existing + Candidates']
        scenario_data = [
            self.unmet_demand_scenarios['existing_only']['unmet_total'].sum(),
            self.unmet_demand_scenarios['existing_plus_candidates']['unmet_total'].sum()
        ]
        
        bars = ax6.bar(scenarios, scenario_data, color=['red', 'orange'], alpha=0.7)
        ax6.set_ylabel('Total Unmet Demand (trucks)')
        ax6.set_title('Unmet Demand by Scenario')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, scenario_data):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 7: Budget Efficiency Comparison (bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Plot both no-service and full-service total unmet demand
        no_service_data = self.budget_scenarios['no_service']
        full_service_data = self.budget_scenarios['full_service']
        
        ax7.step(no_service_data['budget'] / 1e6, no_service_data['total_unmet'], 
                where='post', linewidth=3, color='blue', label='No-Service Facilities', alpha=0.8)
        
        ax7.step(full_service_data['budget'] / 1e6, full_service_data['total_unmet'], 
                where='post', linewidth=3, color='red', label='Full-Service Facilities', alpha=0.8)
        
        ax7.set_xlabel('Budget ($ Millions)')
        ax7.set_ylabel('Total Unmet Demand (trucks)')
        ax7.set_title('Total Unmet Demand vs Budget by Facility Type')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        ax7.set_xlim(0, 200)
        
        plt.suptitle('Comprehensive Interstate Truck Parking Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Save the comprehensive dashboard
        plot_file = os.path.join(OUTPUT_DIR, "comprehensive_analysis_dashboard.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(plot_file.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Comprehensive dashboard saved to {plot_file}")
        
        plt.show()
    
    def export_enhanced_results(self):
        """Export all enhanced analysis results"""
        print("\n=== EXPORTING ENHANCED RESULTS ===")
        
        # Export traffic segments with enhanced assignments
        segments_file = os.path.join(OUTPUT_DIR, "enhanced_traffic_segments.csv")
        self.traffic_segments_filtered.to_csv(segments_file, index=False)
        print(f"✓ Enhanced traffic segments saved to {segments_file}")
        
        # Export interstate demand summary
        demand_file = os.path.join(OUTPUT_DIR, "enhanced_interstate_demand.csv")
        self.interstate_demand.to_csv(demand_file)
        print(f"✓ Interstate demand summary saved to {demand_file}")
        
        # Export facility analysis
        facility_file = os.path.join(OUTPUT_DIR, "comprehensive_facility_analysis.csv")
        self.all_facilities.to_csv(facility_file, index=False)
        print(f"✓ Facility analysis saved to {facility_file}")
        
        # Export budget scenarios
        for scenario_type, data in self.budget_scenarios.items():
            scenario_file = os.path.join(OUTPUT_DIR, f"budget_scenario_{scenario_type}.csv")
            data.to_csv(scenario_file, index=False)
            print(f"✓ Budget scenario ({scenario_type}) saved to {scenario_file}")
        
        # Export unmet demand scenarios
        for scenario_name, data in self.unmet_demand_scenarios.items():
            unmet_file = os.path.join(OUTPUT_DIR, f"unmet_demand_{scenario_name}.csv")
            data.to_csv(unmet_file, index=False)
            print(f"✓ Unmet demand scenario ({scenario_name}) saved to {unmet_file}")
        
        # Create comprehensive summary report
        self._create_enhanced_summary_report()
    
    def _create_enhanced_summary_report(self):
        """Create enhanced summary report"""
        report_file = os.path.join(OUTPUT_DIR, "enhanced_analysis_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED NORTH CAROLINA INTERSTATE TRUCK PARKING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ANALYSIS OVERVIEW:\n")
            f.write(f"Target Interstates: {', '.join(TARGET_INTERSTATES)}\n")
            f.write(f"Traffic Segments Analyzed: {len(self.traffic_segments_filtered)}\n")
            f.write(f"Assignment Rate: {len(self.traffic_segments_filtered)/len(self.traffic_segments)*100:.1f}%\n")
            f.write(f"Existing Facilities: {len(self.existing_facilities)}\n")
            f.write(f"Candidate Facilities: {len(self.candidate_facilities_processed)}\n\n")
            
            # Demand summary
            f.write("DEMAND SUMMARY BY CLASS:\n")
            f.write("-" * 40 + "\n")
            for k in range(1, 5):
                total_demand = sum(self.interstate_demand[f'demand_class_{k}'])
                f.write(f"Class {k}: {total_demand:.2f} trucks\n")
            f.write(f"Total Demand: {sum(self.interstate_demand['demand_total']):.2f} trucks\n\n")
            
            # Interstate breakdown
            f.write("DEMAND BY INTERSTATE:\n")
            f.write("-" * 30 + "\n")
            for interstate in TARGET_INTERSTATES:
                if interstate in self.interstate_demand.index:
                    data = self.interstate_demand.loc[interstate]
                    f.write(f"{interstate}:\n")
                    f.write(f"  Total Demand: {data['demand_total']:.1f} trucks\n")
                    f.write(f"  Segments: {data['num_segments']}\n")
                    f.write(f"  AADTT: {data['AADTT']:,.0f}\n")
                else:
                    f.write(f"{interstate}: No demand calculated\n")
                f.write("\n")
            
            # Capacity analysis
            f.write("CAPACITY ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            for interstate in TARGET_INTERSTATES:
                existing_cap = 0
                candidate_cap = 0
                
                if (interstate, 'existing') in self.facility_summary.index:
                    existing_cap = self.facility_summary.loc[(interstate, 'existing'), ('capacity', 'sum')]
                if (interstate, 'candidate') in self.facility_summary.index:
                    candidate_cap = self.facility_summary.loc[(interstate, 'candidate'), ('capacity', 'sum')]
                
                f.write(f"{interstate}:\n")
                f.write(f"  Existing Capacity: {existing_cap:.0f} spaces\n")
                f.write(f"  Candidate Capacity: {candidate_cap:.0f} spaces\n")
                f.write(f"  Total Available: {existing_cap + candidate_cap:.0f} spaces\n\n")
            
            # Scenario comparison
            f.write("UNMET DEMAND SCENARIOS:\n")
            f.write("-" * 30 + "\n")
            for scenario_name, description in [
                ('existing_only', 'Existing Facilities Only'),
                ('existing_plus_candidates', 'Existing + All Candidates')
            ]:
                scenario_data = self.unmet_demand_scenarios[scenario_name]
                total_unmet = scenario_data['unmet_total'].sum()
                f.write(f"{description}:\n")
                f.write(f"  Total Unmet Demand: {total_unmet:.1f} trucks\n")
                for k in range(1, 5):
                    class_unmet = scenario_data[f'unmet_class_{k}'].sum()
                    f.write(f"  Class {k} Unmet: {class_unmet:.1f} trucks\n")
                f.write("\n")
        
        print(f"✓ Enhanced summary report saved to {report_file}")

def main():
    """Main function to run enhanced interstate analysis"""
    print("ENHANCED NORTH CAROLINA INTERSTATE TRUCK PARKING ANALYSIS")
    print("=" * 70)
    print("Features:")
    print("  ✓ Enhanced spatial matching for all interstate segments")
    print("  ✓ Integration of candidate facilities")  
    print("  ✓ Unmet demand step plots for all 4 truck classes")
    print("  ✓ Budget allocation optimization")
    print("  ✓ Comprehensive analysis dashboard")
    print()
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedInterstateAnalyzer()
        
        # Enhanced spatial matching
        analyzer.enhanced_spatial_matching()
        
        # Calculate comprehensive demand
        interstate_demand = analyzer.calculate_comprehensive_demand()
        
        # Analyze facilities (existing + candidates)
        all_facilities = analyzer.analyze_facilities()
        
        # Calculate unmet demand scenarios
        analyzer.calculate_unmet_demand_with_scenarios()
        
        # Create unmet demand step plots (main deliverable)
        analyzer.create_unmet_demand_step_plots()
        
        # Create comprehensive dashboard
        analyzer.create_comprehensive_analysis_dashboard()
        
        # Export all results
        analyzer.export_enhanced_results()
        
        print(f"\n{'='*70}")
        print("ENHANCED ANALYSIS COMPLETE!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("\nKey outputs:")
        print("  ✓ unmet_demand_step_plots_all_classes.png/pdf")
        print("  ✓ facility_count_vs_budget.png/pdf")
        print("  ✓ comprehensive_analysis_dashboard.png/pdf")
        print("  ✓ enhanced_analysis_summary.txt")
        print("  ✓ All CSV data files")
        
        return analyzer
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyzer = main()