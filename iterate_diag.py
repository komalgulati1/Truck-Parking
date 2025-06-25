"""
CORRECTED UNIFIED FACILITY OPTIMIZATION SYSTEM
Using real traffic data and proper FHWA demand calculation methodology
Removes all random assumptions and uses actual data sources
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pulp
import warnings
import time
from geopy.geocoders import Nominatim
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - REAL DATA PATHS
# =============================================================================

# File paths - REAL DATA PATHS as per your working demand analysis
TRAFFIC_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/results/traffic_segments_interstates_only.gpkg'
INTERSTATE_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/interstate_ncdot.gpkg'
EXISTING_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/existing.gpkg'
CANDIDATE_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/shortlisted_candidates.shp'
COUNTY_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/county.csv'
OUTPUT_DIR = '/Users/komalgulati/Documents/Project_3_1/simulation/results/corrected_optimization'

# Budget scenarios for optimization
BUDGET_SCENARIOS = {
    'current': 175e6,      # Current budget constraint (175 million)
    'expanded': 1000e6     # Expanded budget scenario (1 billion)
}

# Excluded facility types from candidates
EXCLUDED_FACILITY_TYPES = ["DMV", "License", "Welcome Center", "Municipal", "Courthouse", "Rest Area"]

# Weights for demand-based composite scoring
PRIMARY_WEIGHT = 0.6    # Unmet demand reduction (increased from 0.5)
SECONDARY_WEIGHT = 0.4  # Traditional factors (crash, accessibility, etc.)

# Proximity analysis parameters
PROXIMITY_ANALYSIS_RADIUS = 10.0   # Miles - radius for analyzing local infrastructure
TRUCK_STOP_SPACING_THRESHOLD = 20.0  # Miles - minimum spacing between truck stops
REST_AREA_SPACING_THRESHOLD = 5.0   # Miles - minimum spacing between rest areas

# FHWA MILP formulation constants - EXACT VALUES from working demand analysis
F_S = 1.15  # Seasonal peaking factor

P_CLASS = {
    'urban': {'short_haul': 0.36, 'long_haul': 0.64},
    'rural': {'short_haul': 0.07, 'long_haul': 0.93}
}

P_FACILITY = {'rest_area': 0.23, 'truck_stop': 0.77}
P_PEAK = {'short_haul': 0.02, 'long_haul': 0.09}
P_PARK = {'short_haul': 5/60, 'long_haul': 0.783}  # Parking duration in hours

# Urban counties for proper classification
URBAN_COUNTIES = ['Wake', 'Mecklenburg', 'Durham', 'Guilford', 'Forsyth', 'Cumberland', 'Buncombe', 'New Hanover']

# Get the current directory of the script
script_dir = os.path.dirname(__file__)
if not script_dir:
    script_dir = '.'

# =============================================================================
# REAL DATA LOADING FUNCTIONS
# =============================================================================

def load_real_traffic_segments():
    """Load REAL traffic segments with proper demand calculation"""
    print("Loading real traffic segments with FHWA demand calculation...")
    
    try:
        # Load the REAL traffic data
        traffic_gdf = gpd.read_file(TRAFFIC_PATH)
        print(f"✓ Loaded {len(traffic_gdf)} traffic segments from real data")
        
        # Convert to WGS84 if not already
        if traffic_gdf.crs != "EPSG:4326":
            traffic_gdf = traffic_gdf.to_crs(epsg=4326)
            print("✓ Converted CRS to EPSG:4326")
        
        # Calculate midpoint and extract lat/lon
        traffic_gdf['midpoint'] = traffic_gdf.geometry.interpolate(0.5, normalized=True)
        traffic_gdf['Latitude'] = traffic_gdf['midpoint'].y
        traffic_gdf['Longitude'] = traffic_gdf['midpoint'].x
        print("✓ Extracted midpoint coordinates")
        
        # Convert to DataFrame and keep relevant columns
        traffic_segments = pd.DataFrame(traffic_gdf.drop(columns='geometry'))
        
        # Verify AADTT column exists (truck traffic field)
        if 'AADTT' not in traffic_segments.columns:
            print(f"ERROR: AADTT column not found. Available columns: {list(traffic_segments.columns)}")
            return None
        
        print(f"✓ AADTT range: {traffic_segments['AADTT'].min():.0f} to {traffic_segments['AADTT'].max():.0f}")
        print(f"✓ Coordinates range: Lat {traffic_segments['Latitude'].min():.3f}-{traffic_segments['Latitude'].max():.3f}, "
              f"Lon {traffic_segments['Longitude'].min():.3f}-{traffic_segments['Longitude'].max():.3f}")
        
        # Assign counties using RouteID mapping
        traffic_segments = assign_counties_by_route_id(traffic_segments)
        
        # Use existing interstate_assignment field
        traffic_segments = assign_interstate_from_data(traffic_segments)
        
        # Calculate FHWA demand using real methodology
        traffic_segments = calculate_fhwa_demand(traffic_segments)
        
        return traffic_segments
        
    except Exception as e:
        print(f"ERROR loading traffic segments: {e}")
        return None

def assign_counties_by_route_id(traffic_segments):
    """Assign counties using RouteID mapping - EXACT method from working code"""
    print("Assigning counties by RouteID...")
    
    # NC county code mapping dictionary - EXACT from working code
    nc_county_codes = {
        '001': 'Alamance', '002': 'Alexander', '003': 'Alleghany', '004': 'Anson', '005': 'Ashe',
        '006': 'Avery', '007': 'Beaufort', '008': 'Bertie', '009': 'Bladen', '010': 'Brunswick',
        '011': 'Buncombe', '012': 'Burke', '013': 'Cabarrus', '014': 'Caldwell', '015': 'Camden',
        '016': 'Carteret', '017': 'Caswell', '018': 'Catawba', '019': 'Chatham', '020': 'Cherokee',
        '021': 'Chowan', '022': 'Clay', '023': 'Cleveland', '024': 'Columbus', '025': 'Craven',
        '026': 'Cumberland', '027': 'Currituck', '028': 'Dare', '029': 'Davidson', '030': 'Davie',
        '031': 'Duplin', '032': 'Durham', '033': 'Edgecombe', '034': 'Forsyth', '035': 'Franklin',
        '036': 'Gaston', '037': 'Gates', '038': 'Graham', '039': 'Granville', '040': 'Greene',
        '041': 'Guilford', '042': 'Halifax', '043': 'Harnett', '044': 'Haywood', '045': 'Henderson',
        '046': 'Hertford', '047': 'Hoke', '048': 'Hyde', '049': 'Iredell', '050': 'Jackson',
        '051': 'Johnston', '052': 'Jones', '053': 'Lee', '054': 'Lenoir', '055': 'Lincoln',
        '056': 'Macon', '057': 'Madison', '058': 'Martin', '059': 'McDowell', '060': 'Mecklenburg',
        '061': 'Mitchell', '062': 'Montgomery', '063': 'Moore', '064': 'Nash', '065': 'New Hanover',
        '066': 'Northampton', '067': 'Onslow', '068': 'Orange', '069': 'Pamlico', '070': 'Pasquotank',
        '071': 'Pender', '072': 'Perquimans', '073': 'Person', '074': 'Pitt', '075': 'Polk',
        '076': 'Randolph', '077': 'Richmond', '078': 'Robeson', '079': 'Rockingham', '080': 'Rowan',
        '081': 'Rutherford', '082': 'Sampson', '083': 'Scotland', '084': 'Stanly', '085': 'Stokes',
        '086': 'Surry', '087': 'Swain', '088': 'Transylvania', '089': 'Tyrrell', '090': 'Union',
        '091': 'Vance', '092': 'Wake', '093': 'Warren', '094': 'Washington', '095': 'Watauga',
        '096': 'Wayne', '097': 'Wilkes', '098': 'Wilson', '099': 'Yadkin', '100': 'Yancey'
    }
    
    def assign_county_by_route_code(row):
        route_id = str(row['RouteID'])
        
        if len(route_id) >= 3:
            last_three = route_id[-3:]
            if last_three in nc_county_codes:
                return nc_county_codes[last_three]
            if last_three.startswith('0'):
                padded = last_three[1:].zfill(3)
                if padded in nc_county_codes:
                    return nc_county_codes[padded]
        
        if len(route_id) >= 2:
            last_two = route_id[-2:].zfill(3)
            if last_two in nc_county_codes:
                return nc_county_codes[last_two]
        
        return 'Unknown'
    
    # Apply county assignment
    traffic_segments['assigned_county'] = traffic_segments.apply(assign_county_by_route_code, axis=1)
    
    # Report results
    county_counts = traffic_segments['assigned_county'].value_counts()
    known_counties = county_counts[county_counts.index != 'Unknown']
    unknown_count = county_counts.get('Unknown', 0)
    
    print(f"✓ County assignment complete:")
    print(f"  Successfully assigned: {len(known_counties)} different counties")
    print(f"  Unknown assignments: {unknown_count} segments")
    print(f"  Top counties: {dict(known_counties.head(5))}")
    
    return traffic_segments

def assign_interstate_from_data(traffic_segments):
    """Use existing interstate_assignment field from the data"""
    print("Using interstate_assignment field from data...")
    
    if 'interstate_assignment' not in traffic_segments.columns:
        print("ERROR: interstate_assignment field not found in traffic segments")
        return traffic_segments
    
    # Report interstate distribution
    interstate_counts = traffic_segments['interstate_assignment'].value_counts()
    print(f"✓ Interstate distribution from data:")
    for interstate, count in interstate_counts.items():
        print(f"  {interstate}: {count} segments")
    
    return traffic_segments

def calculate_fhwa_demand(traffic_segments):
    """Calculate demand using EXACT FHWA MILP formula from working code"""
    print("Calculating FHWA demand for all 4 classes...")
    
    # Add urban/rural classification
    traffic_segments['is_urban'] = traffic_segments['assigned_county'].isin(URBAN_COUNTIES)
    
    # Calculate segment properties
    if 'EndMP' in traffic_segments.columns and 'BeginMP' in traffic_segments.columns:
        traffic_segments['length_miles'] = (
            traffic_segments['EndMP'] - traffic_segments['BeginMP']
        ).clip(lower=0.1, upper=20.0)
    else:
        # Use default length if milepost data not available
        traffic_segments['length_miles'] = 1.0
        print("WARNING: Using default segment length (EndMP/BeginMP not found)")
    
    traffic_segments['speed_limit'] = 60  # Default assumption for interstates
    traffic_segments['travel_time_hours'] = (
        traffic_segments['length_miles'] / traffic_segments['speed_limit']
    )
    
    # Define truck classes - EXACT from working code
    truck_classes = {
        1: {'name': 'Short-Haul Rest Area', 'haul': 'short_haul', 'facility': 'rest_area'},
        2: {'name': 'Short-Haul Truck Stop', 'haul': 'short_haul', 'facility': 'truck_stop'},
        3: {'name': 'Long-Haul Rest Area', 'haul': 'long_haul', 'facility': 'rest_area'},
        4: {'name': 'Long-Haul Truck Stop', 'haul': 'long_haul', 'facility': 'truck_stop'}
    }
    
    print("Applying FHWA MILP formula step by step:")
    print(f"  F_S (Seasonal factor): {F_S}")
    print(f"  Urban vs Rural ratios: {P_CLASS}")
    print(f"  Facility preferences: {P_FACILITY}")
    print(f"  Peak factors: {P_PEAK}")
    print(f"  Parking durations: {P_PARK}")
    
    # Calculate demand for each class using EXACT FHWA formula
    for k, class_info in truck_classes.items():
        haul_type = class_info['haul']
        facility_type = class_info['facility']
        
        print(f"  Calculating Class {k}: {class_info['name']}")
        
        # Step-by-step FHWA calculation using AADTT (truck traffic)
        base_aadtt = traffic_segments['AADTT']
        seasonal_aadtt = base_aadtt * F_S
        
        area_type = traffic_segments['is_urban'].map({True: 'urban', False: 'rural'})
        class_proportion = area_type.map(lambda x: P_CLASS[x][haul_type])
        class_traffic = seasonal_aadtt * class_proportion
        
        facility_traffic = class_traffic * P_FACILITY[facility_type]
        peak_traffic = facility_traffic * P_PEAK[haul_type]
        travel_time_factor = traffic_segments['travel_time_hours']
        parking_factor = P_PARK[haul_type]
        
        final_demand = peak_traffic * travel_time_factor * parking_factor
        
        # Store results
        traffic_segments[f'demand_class_{k}'] = final_demand.fillna(0)
        
        print(f"    Total demand: {final_demand.sum():.2f} trucks")
    
    # Calculate aggregated demands
    traffic_segments['demand_short_haul'] = (
        traffic_segments['demand_class_1'] + traffic_segments['demand_class_2']
    )
    traffic_segments['demand_long_haul'] = (
        traffic_segments['demand_class_3'] + traffic_segments['demand_class_4']
    )
    traffic_segments['demand_total'] = (
        traffic_segments['demand_short_haul'] + traffic_segments['demand_long_haul']
    )
    
    print(f"✓ Total demand across all classes: {traffic_segments['demand_total'].sum():.2f} trucks")
    
    return traffic_segments

def get_demand_class_descriptions():
    """Return descriptions of demand classes for reporting"""
    return {
        'class_1': 'Short-Haul Rest Area (basic service, day trips)',
        'class_2': 'Short-Haul Truck Stop (full service, day trips)', 
        'class_3': 'Long-Haul Rest Area (basic service, overnight)',
        'class_4': 'Long-Haul Truck Stop (full service, overnight)'
    }

# =============================================================================
# UNMET DEMAND CALCULATOR - CORRECTED VERSION
# =============================================================================

class UnmetDemandCalculator:
    """Calculate unmet demand using Linear Programming with real traffic data"""
    
    def __init__(self, traffic_segments, facilities, base_distance_threshold=5.0):
        self.traffic_segments = traffic_segments.copy()
        self.facilities = facilities.copy()
        self.base_distance_threshold = base_distance_threshold
        
        print(f"Initializing demand calculator with {len(traffic_segments)} segments and {len(facilities)} facilities")
        
        # Validate coordinates
        self._validate_coordinates()
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
        print("Demand calculator initialization completed successfully")
    
    def _validate_coordinates(self):
        """Validate that coordinates are in proper geographic format"""
        seg_lats = self.traffic_segments['Latitude'].values
        seg_lons = self.traffic_segments['Longitude'].values
        fac_lats = self.facilities['Latitude'].values
        fac_lons = self.facilities['Longitude'].values
        
        # Check for reasonable coordinate ranges
        seg_lat_range = (seg_lats.min(), seg_lats.max())
        seg_lon_range = (seg_lons.min(), seg_lons.max())
        fac_lat_range = (fac_lats.min(), fac_lats.max())
        fac_lon_range = (fac_lons.min(), fac_lons.max())
        
        print(f"Coordinate validation:")
        print(f"  Segments - Lat: {seg_lat_range}, Lon: {seg_lon_range}")
        print(f"  Facilities - Lat: {fac_lat_range}, Lon: {fac_lon_range}")
        
        # Warning for projected coordinates
        if (abs(seg_lats).max() > 180 or abs(seg_lons).max() > 180 or 
            abs(fac_lats).max() > 180 or abs(fac_lons).max() > 180):
            print("WARNING: Coordinates appear to be projected, not geographic!")
    
    def _calculate_distance_matrix(self):
        """Calculate distance matrix between segments and facilities"""
        n_segments = len(self.traffic_segments)
        n_facilities = len(self.facilities)
        
        seg_lats = self.traffic_segments['Latitude'].values
        seg_lons = self.traffic_segments['Longitude'].values
        fac_lats = self.facilities['Latitude'].values
        fac_lons = self.facilities['Longitude'].values
        
        distances = np.zeros((n_segments, n_facilities))
        valid_count = 0
        
        for i in range(n_segments):
            for j in range(n_facilities):
                dist = self._haversine_distance(
                    seg_lats[i], seg_lons[i], fac_lats[j], fac_lons[j]
                )
                distances[i, j] = dist
                if not np.isnan(dist):
                    valid_count += 1
        
        print(f"Distance matrix calculated: {valid_count}/{n_segments * n_facilities} valid distances")
        
        if valid_count > 0:
            valid_dists = distances[~np.isnan(distances)]
            print(f"Distance range: {valid_dists.min():.1f} to {valid_dists.max():.1f} miles")
        
        return distances
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points in miles"""
        R = 3959.87433  # Earth radius in miles
        
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return np.nan
        
        if (abs(lat1) > 90 or abs(lat2) > 90 or 
            abs(lon1) > 180 or abs(lon2) > 180):
            return np.nan
        
        try:
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        except:
            return np.nan
    
    def calculate_unmet_demand_lp(self, selected_facilities, facility_types):
        """Calculate unmet demand using LP optimization"""
        classes = {
            1: {'name': 'Short-Haul Rest Area', 'haul': 'short_haul', 'facility': 'rest_area'},
            2: {'name': 'Short-Haul Truck Stop', 'haul': 'short_haul', 'facility': 'truck_stop'},
            3: {'name': 'Long-Haul Rest Area', 'haul': 'long_haul', 'facility': 'rest_area'},
            4: {'name': 'Long-Haul Truck Stop', 'haul': 'long_haul', 'facility': 'truck_stop'}
        }
        
        unmet_results = {}
        n_segments = len(self.traffic_segments)
        
        for class_id, class_info in classes.items():
            demand_col = f'demand_class_{class_id}'
            
            if demand_col not in self.traffic_segments.columns:
                unmet_results[f'unmet_class_{class_id}'] = 0
                continue
                
            segment_demands = self.traffic_segments[demand_col].values
            
            # Filter compatible facilities
            compatible_facilities = [
                j for j in selected_facilities 
                if facility_types.get(j, '') == class_info['facility']
            ]
            
            if not compatible_facilities:
                unmet_results[f'unmet_class_{class_id}'] = segment_demands.sum()
                continue
            
            try:
                prob = pulp.LpProblem(f"Class_{class_id}_Unmet_Demand", pulp.LpMinimize)
                y_vars = {}
                u_vars = {}
                
                # Decision variables
                for i in range(n_segments):
                    u_vars[i] = pulp.LpVariable(f"u_{i}", 0, None)
                    
                    for j in compatible_facilities:
                        if (not np.isnan(self.distance_matrix[i, j]) and 
                            self.distance_matrix[i, j] <= self.base_distance_threshold):
                            y_vars[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", 0, None)
                
                # Objective: minimize total unmet demand
                prob += pulp.lpSum([u_vars[i] for i in range(n_segments)])
                
                # Demand satisfaction constraints
                for i in range(n_segments):
                    prob += (
                        pulp.lpSum([y_vars.get((i, j), 0) for j in compatible_facilities if (i, j) in y_vars])
                        + u_vars[i] == segment_demands[i]
                    )
                
                # Capacity constraints
                for j in compatible_facilities:
                    cap = self.facilities.iloc[j]['capacity']
                    if cap > 0:
                        prob += (
                            pulp.lpSum([y_vars.get((i, j), 0) for i in range(n_segments) if (i, j) in y_vars])
                            <= cap
                        )
                
                # Solve the optimization problem
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                
                if prob.status == pulp.LpStatusOptimal:
                    unmet_total = sum([u_vars[i].varValue or 0 for i in range(n_segments)])
                    unmet_results[f'unmet_class_{class_id}'] = max(0, unmet_total)
                else:
                    unmet_results[f'unmet_class_{class_id}'] = segment_demands.sum()
                    
            except Exception as e:
                print(f"Error in LP calculation for class {class_id}: {e}")
                unmet_results[f'unmet_class_{class_id}'] = segment_demands.sum()
        
        # Aggregate metrics
        unmet_results['unmet_short_haul'] = unmet_results.get('unmet_class_1', 0) + unmet_results.get('unmet_class_2', 0)
        unmet_results['unmet_long_haul'] = unmet_results.get('unmet_class_3', 0) + unmet_results.get('unmet_class_4', 0)
        unmet_results['unmet_total'] = unmet_results['unmet_short_haul'] + unmet_results['unmet_long_haul']
        
        return unmet_results

# =============================================================================
# PROXIMITY ANALYSIS CLASS
# =============================================================================

class ProximityAnalyzer:
    """Intelligent proximity analysis for facility type selection"""
    
    def __init__(self, all_facilities):
        self.all_facilities = all_facilities
        self.facility_coordinates = self._extract_coordinates()
    
    def _extract_coordinates(self):
        """Extract coordinate arrays for efficient distance calculations"""
        return {
            'latitudes': self.all_facilities['Latitude'].values,
            'longitudes': self.all_facilities['Longitude'].values
        }
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points in miles"""
        R = 3959.87433  # Earth radius in miles
        
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return np.nan
        
        try:
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        except:
            return np.nan
    
    def analyze_local_infrastructure(self, candidate_lat, candidate_lon, current_facility_types):
        """Analyze local infrastructure around a candidate location"""
        
        # Calculate distances to all existing and selected facilities
        distances_and_types = []
        
        for idx, row in self.all_facilities.iterrows():
            if idx in current_facility_types:  # Only consider facilities that are built or selected
                distance = self._haversine_distance(
                    candidate_lat, candidate_lon,
                    row['Latitude'], row['Longitude']
                )
                
                if not np.isnan(distance) and distance <= PROXIMITY_ANALYSIS_RADIUS:
                    facility_type = current_facility_types[idx]
                    distances_and_types.append({
                        'distance': distance,
                        'facility_type': facility_type,
                        'facility_idx': idx
                    })
        
        # Sort by distance to find nearest facilities
        distances_and_types.sort(key=lambda x: x['distance'])
        
        # Analyze local infrastructure composition
        nearby_truck_stops = [f for f in distances_and_types if f['facility_type'] == 'truck_stop']
        nearby_rest_areas = [f for f in distances_and_types if f['facility_type'] == 'rest_area']
        
        # Strategic facility type recommendation
        recommendation = self._determine_facility_type_recommendation(
            distances_and_types, nearby_truck_stops, nearby_rest_areas
        )
        
        return recommendation
    
    def _determine_facility_type_recommendation(self, nearby_facilities, nearby_truck_stops, nearby_rest_areas):
        """Apply strategic logic to recommend facility type"""
        
        # If no nearby facilities, prefer truck stop
        if not nearby_facilities:
            return {
                'recommended_type': 'truck_stop',
                'confidence': 0.8,
                'reasoning': 'No nearby facilities - truck stop provides comprehensive coverage'
            }
        
        # Check for very close truck stops
        closest_truck_stop_distance = min([f['distance'] for f in nearby_truck_stops]) if nearby_truck_stops else float('inf')
        
        if closest_truck_stop_distance < TRUCK_STOP_SPACING_THRESHOLD:
            return {
                'recommended_type': 'rest_area',
                'confidence': 0.9,
                'reasoning': f'Truck stop within {closest_truck_stop_distance:.1f} miles - rest area provides complementary coverage'
            }
        
        # Check for rest area saturation
        closest_rest_area_distance = min([f['distance'] for f in nearby_rest_areas]) if nearby_rest_areas else float('inf')
        
        if closest_rest_area_distance < REST_AREA_SPACING_THRESHOLD and closest_truck_stop_distance > TRUCK_STOP_SPACING_THRESHOLD:
            return {
                'recommended_type': 'truck_stop',
                'confidence': 0.8,
                'reasoning': f'Rest area within {closest_rest_area_distance:.1f} miles but no close truck stops - truck stop fills service gap'
            }
        
        # Balance assessment
        truck_stop_density = len([f for f in nearby_truck_stops if f['distance'] <= 20])
        rest_area_density = len([f for f in nearby_rest_areas if f['distance'] <= 20])
        
        if truck_stop_density >= 2 and rest_area_density == 0:
            return {
                'recommended_type': 'rest_area',
                'confidence': 0.8,
                'reasoning': f'Multiple truck stops ({truck_stop_density}) but no rest areas - rest area fills basic service gap'
            }
        elif rest_area_density >= 2 and truck_stop_density == 0:
            return {
                'recommended_type': 'truck_stop',
                'confidence': 0.8,
                'reasoning': f'Multiple rest areas ({rest_area_density}) but no truck stops - truck stop fills full service gap'
            }
        elif truck_stop_density > rest_area_density + 1:
            return {
                'recommended_type': 'rest_area',
                'confidence': 0.7,
                'reasoning': f'Local area has {truck_stop_density} truck stops vs {rest_area_density} rest areas - rest area balances coverage'
            }
        else:
            return {
                'recommended_type': 'truck_stop',
                'confidence': 0.7,
                'reasoning': f'Local area needs comprehensive service - truck stop provides full amenities'
            }

# =============================================================================
# DEMAND-BASED FACILITY TYPE ANALYSIS
# =============================================================================

def analyze_demand_patterns(candidate_lat, candidate_lon, traffic_segments, radius=5.0):
    """Analyze demand patterns around a candidate location to recommend facility type"""
    
    # Calculate distances to all traffic segments
    segment_distances = []
    
    for idx, segment in traffic_segments.iterrows():
        # Haversine distance calculation
        lat_diff = np.radians(candidate_lat - segment['Latitude'])
        lon_diff = np.radians(candidate_lon - segment['Longitude'])
        lat1, lat2 = np.radians(segment['Latitude']), np.radians(candidate_lat)
        
        a = np.sin(lat_diff/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon_diff/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 3959.87433 * c  # Earth radius in miles
        
        if distance <= radius:
            segment_distances.append({
                'distance': distance,
                'demand_class_1': segment['demand_class_1'],  # Rest area short-haul
                'demand_class_2': segment['demand_class_2'],  # Truck stop short-haul  
                'demand_class_3': segment['demand_class_3'],  # Rest area long-haul
                'demand_class_4': segment['demand_class_4']   # Truck stop long-haul
            })
    
    if not segment_distances:
        return {
            'recommended_type': 'truck_stop',
            'confidence': 0.6,
            'reasoning': 'No nearby demand data - truck stop provides comprehensive coverage',
            'demand_analysis': {'rest_area_total': 0, 'truck_stop_total': 0, 'all_classes': [0, 0, 0, 0]}
        }
    
    # Calculate weighted demand for ALL 4 classes individually
    class_demands = [0, 0, 0, 0]  # Classes 1, 2, 3, 4
    
    for segment in segment_distances:
        weight = 1 / (1 + segment['distance'])  # Inverse distance weighting
        
        class_demands[0] += segment['demand_class_1'] * weight
        class_demands[1] += segment['demand_class_2'] * weight
        class_demands[2] += segment['demand_class_3'] * weight
        class_demands[3] += segment['demand_class_4'] * weight
    
    # Aggregate by facility type
    rest_area_demand = class_demands[0] + class_demands[2]    # Classes 1 + 3
    truck_stop_demand = class_demands[1] + class_demands[3]   # Classes 2 + 4
    total_demand = rest_area_demand + truck_stop_demand
    
    if total_demand == 0:
        return {
            'recommended_type': 'truck_stop',
            'confidence': 0.5,
            'reasoning': 'No significant demand detected - truck stop for general coverage',
            'demand_analysis': {'rest_area_total': 0, 'truck_stop_total': 0, 'all_classes': class_demands}
        }
    
    # Decision logic based on demand ratios and significance
    truck_stop_ratio = truck_stop_demand / total_demand
    
    if truck_stop_ratio > 0.7:
        return {
            'recommended_type': 'truck_stop',
            'confidence': min(0.9, truck_stop_ratio + 0.2),
            'reasoning': f'Strong truck stop demand ({truck_stop_ratio:.1%})',
            'demand_analysis': {
                'rest_area_total': rest_area_demand,
                'truck_stop_total': truck_stop_demand,
                'all_classes': class_demands
            }
        }
    elif truck_stop_ratio < 0.3:
        return {
            'recommended_type': 'rest_area',
            'confidence': min(0.9, (1 - truck_stop_ratio) + 0.2),
            'reasoning': f'Strong rest area demand ({(1-truck_stop_ratio):.1%})',
            'demand_analysis': {
                'rest_area_total': rest_area_demand,
                'truck_stop_total': truck_stop_demand,
                'all_classes': class_demands
            }
        }
    else:
        # Balanced demand - consider cost effectiveness
        if rest_area_demand > truck_stop_demand * 0.4:  # Rest area demand is significant
            return {
                'recommended_type': 'rest_area',
                'confidence': 0.6,
                'reasoning': f'Balanced demand - rest area cost-effective',
                'demand_analysis': {
                    'rest_area_total': rest_area_demand,
                    'truck_stop_total': truck_stop_demand,
                    'all_classes': class_demands
                }
            }
        else:
            return {
                'recommended_type': 'truck_stop',
                'confidence': 0.6,
                'reasoning': f'Balanced demand - truck stop for comprehensive service',
                'demand_analysis': {
                    'rest_area_total': rest_area_demand,
                    'truck_stop_total': truck_stop_demand,
                    'all_classes': class_demands
                }
            }

# =============================================================================
# LOCATION INFORMATION FUNCTIONS
# =============================================================================

def load_county_data():
    """Load county boundary data for North Carolina"""
    print("Checking for county boundary data...")
    
    county_paths = [
        os.path.join(script_dir, "../datasets/nc_counties.shp"),
        os.path.join(script_dir, "../datasets/counties.shp"),
        os.path.join(script_dir, "../datasets/NC_counties.shp")
    ]
    
    for path in county_paths:
        try:
            counties_gdf = gpd.read_file(path)
            print(f"✓ Loaded county boundary data from: {path}")
            if counties_gdf.crs != "EPSG:4326":
                counties_gdf = counties_gdf.to_crs("EPSG:4326")
            return counties_gdf
        except:
            continue
    
    print("ℹ️  County boundary shapefiles not found - using RouteID-based county assignment")
    return None

def get_location_info(lat, lon, counties_gdf=None):
    """Get county/city information for a given coordinate"""
    if counties_gdf is not None:
        try:
            point = Point(lon, lat)
            
            for idx, county in counties_gdf.iterrows():
                if county.geometry.contains(point):
                    county_name = county.get('NAME', county.get('COUNTY', county.get('name', 'Unknown')))
                    return {
                        'county': county_name,
                        'city': 'Unknown',
                        'method': 'spatial_join'
                    }
        except Exception as e:
            print(f"Error in spatial join: {e}")
    
    # Fallback to reverse geocoding
    try:
        geolocator = Nominatim(user_agent="facility_optimization")
        location = geolocator.reverse((lat, lon), timeout=10)
        
        if location and location.raw.get('address'):
            address = location.raw['address']
            county = address.get('county', 'Unknown')
            city = address.get('city', address.get('town', address.get('village', 'Unknown')))
            
            return {
                'county': county,
                'city': city,
                'method': 'reverse_geocoding'
            }
    except Exception as e:
        print(f"Error in reverse geocoding for {lat}, {lon}: {e}")
    
    return get_nc_county_estimate(lat, lon)

def get_nc_county_estimate(lat, lon):
    """Rough county estimation for North Carolina based on coordinate ranges"""
    nc_counties = {
        'Mecklenburg': {'lat_range': (35.0, 35.5), 'lon_range': (-81.1, -80.5)},
        'Wake': {'lat_range': (35.6, 36.0), 'lon_range': (-78.9, -78.3)},
        'Guilford': {'lat_range': (35.9, 36.3), 'lon_range': (-80.1, -79.6)},
        'Forsyth': {'lat_range': (36.0, 36.3), 'lon_range': (-80.4, -80.0)},
        'Durham': {'lat_range': (35.8, 36.1), 'lon_range': (-79.1, -78.7)},
        'Cumberland': {'lat_range': (34.9, 35.4), 'lon_range': (-79.2, -78.6)},
        'New Hanover': {'lat_range': (34.1, 34.4), 'lon_range': (-78.1, -77.7)},
    }
    
    for county, ranges in nc_counties.items():
        if (ranges['lat_range'][0] <= lat <= ranges['lat_range'][1] and 
            ranges['lon_range'][0] <= lon <= ranges['lon_range'][1]):
            return {
                'county': county,
                'city': 'Unknown',
                'method': 'coordinate_estimate'
            }
    
    return {
        'county': 'Unknown NC County',
        'city': 'Unknown',
        'method': 'fallback'
    }

def add_location_info_to_candidates(candidates_gdf, counties_gdf):
    """Add county and city information to candidate facilities"""
    print("Adding location information to candidate facilities...")
    
    counties = []
    cities = []
    location_methods = []
    
    for idx, candidate in candidates_gdf.iterrows():
        try:
            lat = candidate.geometry.y
            lon = candidate.geometry.x
            
            location_info = get_location_info(lat, lon, counties_gdf)
            
            counties.append(location_info['county'])
            cities.append(location_info['city'])
            location_methods.append(location_info['method'])
            
            if location_info['method'] == 'reverse_geocoding':
                time.sleep(0.1)  # Respectful delay
                
        except Exception as e:
            print(f"Error processing candidate {idx}: {e}")
            counties.append('Unknown')
            cities.append('Unknown')
            location_methods.append('error')
    
    candidates_gdf['county'] = counties
    candidates_gdf['city'] = cities
    candidates_gdf['location_method'] = location_methods
    
    print(f"Location information added to {len(candidates_gdf)} candidates")
    
    method_counts = pd.Series(location_methods).value_counts()
    print(f"Location determination methods: {dict(method_counts)}")
    
    return candidates_gdf

# =============================================================================
# COST CALCULATION FUNCTIONS
# =============================================================================

def calculate_development_costs(candidates_gdf):
    """Calculate development costs with realistic capacity caps"""
    print("Calculating development costs with capacity constraints...")
    
    if 'capacity_value' not in candidates_gdf.columns:
        raise ValueError("'capacity_value' field not found in candidates data")
    
    # Cost model parameters
    SITE_PREP_NO_SERVICE = 200000      # $200K site preparation for basic facilities
    COST_PER_SPACE_NO_SERVICE = 10000  # $10K per parking space for basic facilities
    
    SITE_PREP_FULL_SERVICE = 7000000   # $7M site preparation for full-service facilities
    COST_PER_SPACE_FULL_SERVICE = 67000 # $67K per space for full-service facilities
    
    # Maximum project values
    MAX_NO_SERVICE_COST = 1100000      # $1.1M maximum for rest areas
    MAX_FULL_SERVICE_COST = 13000000   # $13M maximum for truck stops
    
    # Calculate maximum feasible capacity
    max_capacity_no_service = (MAX_NO_SERVICE_COST - SITE_PREP_NO_SERVICE) // COST_PER_SPACE_NO_SERVICE
    max_capacity_full_service = (MAX_FULL_SERVICE_COST - SITE_PREP_FULL_SERVICE) // COST_PER_SPACE_FULL_SERVICE
    
    print(f"Capacity constraints:")
    print(f"  Basic facilities: maximum {max_capacity_no_service} spaces")
    print(f"  Full-service facilities: maximum {max_capacity_full_service} spaces")
    
    # Apply capacity caps
    candidates_gdf['capped_capacity_no_service'] = candidates_gdf['capacity_value'].clip(upper=max_capacity_no_service)
    candidates_gdf['capped_capacity_full_service'] = candidates_gdf['capacity_value'].clip(upper=max_capacity_full_service)
    
    # Calculate costs
    candidates_gdf['cost_no_service'] = (SITE_PREP_NO_SERVICE + 
                                       candidates_gdf['capped_capacity_no_service'] * COST_PER_SPACE_NO_SERVICE)
    
    candidates_gdf['cost_full_service'] = (SITE_PREP_FULL_SERVICE + 
                                         candidates_gdf['capped_capacity_full_service'] * COST_PER_SPACE_FULL_SERVICE)
    
    # Verify costs don't exceed maximum
    assert candidates_gdf['cost_no_service'].max() <= MAX_NO_SERVICE_COST, "No-service costs exceed maximum!"
    assert candidates_gdf['cost_full_service'].max() <= MAX_FULL_SERVICE_COST, "Full-service costs exceed maximum!"
    
    # Report adjustments
    capacity_reduced_basic = (candidates_gdf['capacity_value'] > max_capacity_no_service).sum()
    capacity_reduced_full = (candidates_gdf['capacity_value'] > max_capacity_full_service).sum()
    
    print(f"Capacity adjustments:")
    print(f"  {capacity_reduced_basic} sites reduced for basic facilities")
    print(f"  {capacity_reduced_full} sites reduced for full-service facilities")
    
    print(f"Cost ranges:")
    print(f"  Basic: ${candidates_gdf['cost_no_service'].min()/1e6:.2f}M to ${candidates_gdf['cost_no_service'].max()/1e6:.2f}M")
    print(f"  Full-service: ${candidates_gdf['cost_full_service'].min()/1e6:.2f}M to ${candidates_gdf['cost_full_service'].max()/1e6:.2f}M")
    
    return candidates_gdf

# =============================================================================
# EXISTING FACILITIES PROCESSING
# =============================================================================

def smart_assign_existing_facility_types(existing_facilities):
    """Enhanced facility type assignment that properly maps real-world facility types"""
    facility_types = {}
    excluded_facilities = []
    
    print("=== ENHANCED EXISTING FACILITY TYPE ASSIGNMENT ===")
    
    has_overnight = 'over_night' in existing_facilities.columns
    has_facility_t = 'Facility_T' in existing_facilities.columns
    
    print(f"Available data columns: over_night={has_overnight}, Facility_T={has_facility_t}")
    
    if not has_facility_t:
        print("WARNING: Facility_T column missing. Excluding all existing facilities.")
        return {}, list(existing_facilities.index)
    
    type_counts = {'rest_area': 0, 'truck_stop': 0}
    
    for idx, facility in existing_facilities.iterrows():
        facility_t = facility.get('Facility_T', None)
        
        if pd.isna(facility_t) or facility_t == '':
            excluded_facilities.append(idx)
            continue
        
        facility_t_str = str(facility_t).lower()
        
        # Public facilities = rest areas
        if facility_t_str == 'public':
            facility_types[idx] = 'rest_area'
            type_counts['rest_area'] += 1
            
        # Walmart facilities: context-dependent
        elif facility_t_str == 'wal-mart':
            if has_overnight:
                over_night = facility.get('over_night', None)
                if pd.isna(over_night) or str(over_night).lower() != 'yes':
                    facility_types[idx] = 'rest_area'
                    type_counts['rest_area'] += 1
                else:
                    facility_types[idx] = 'truck_stop'
                    type_counts['truck_stop'] += 1
            else:
                facility_types[idx] = 'rest_area'
                type_counts['rest_area'] += 1
                
        # All other private facilities = truck stops
        else:
            facility_types[idx] = 'truck_stop'
            type_counts['truck_stop'] += 1
    
    print(f"Facility type assignment results:")
    print(f"  Rest areas: {type_counts['rest_area']}")
    print(f"  Truck stops: {type_counts['truck_stop']}")
    print(f"  Excluded: {len(excluded_facilities)}")
    
    return facility_types, excluded_facilities

def prepare_facilities_for_optimization(candidates_gdf, existing_facilities, traffic_segments):
    """Prepare unified facility dataset for optimization"""
    print("Preparing facilities for optimization...")
    
    # Smart assignment of existing facility types
    existing_facility_types, excluded_facilities = smart_assign_existing_facility_types(existing_facilities)
    
    # Filter out excluded existing facilities
    existing_clean = existing_facilities.drop(excluded_facilities).copy()
    print(f"Using {len(existing_clean)} existing facilities after filtering")
    
    # Process existing facilities
    existing_clean['capacity'] = existing_clean.get('Final_Park', 0).fillna(0)
    existing_clean['facility_type'] = 'existing'
    
    # Extract coordinates
    try:
        existing_clean['Latitude'] = existing_clean.geometry.y
        existing_clean['Longitude'] = existing_clean.geometry.x
    except Exception as e:
        print(f"Error extracting existing facility coordinates: {e}")
        coords = existing_clean.geometry.apply(lambda geom: (geom.x, geom.y) if hasattr(geom, 'x') else (None, None))
        existing_clean['Longitude'] = coords.apply(lambda x: x[0])
        existing_clean['Latitude'] = coords.apply(lambda x: x[1])
    
    # Create existing facility records
    existing_records = []
    old_to_new_index_mapping = {}
    
    for new_idx, (old_idx, row) in enumerate(existing_clean.iterrows()):
        existing_records.append({
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'capacity': row['capacity'],
            'facility_type': 'existing',
            'original_index': old_idx
        })
        old_to_new_index_mapping[old_idx] = new_idx
    
    # Process candidate facilities
    candidate_records = []
    candidate_start_idx = len(existing_records)
    
    print(f"Processing {len(candidates_gdf)} candidate facilities...")
    
    for idx, row in candidates_gdf.iterrows():
        try:
            lat, lon = row.geometry.y, row.geometry.x
            
            candidate_records.append({
                'Latitude': lat,
                'Longitude': lon,
                'capacity': row['capacity_value'],
                'facility_type': 'candidate',
                'original_index': idx
            })
        except Exception as e:
            print(f"Error processing candidate {idx}: {e}")
            continue
    
    # Combine all facilities
    all_facilities_list = existing_records + candidate_records
    all_facilities = pd.DataFrame(all_facilities_list)
    
    print(f"Combined facilities dataset:")
    print(f"  Existing: {len(existing_records)}")
    print(f"  Candidates: {len(candidate_records)}")
    print(f"  Total: {len(all_facilities)}")
    
    # Create facility type mapping
    facility_type_mapping = {}
    for new_idx, row in all_facilities.iterrows():
        if row['facility_type'] == 'existing':
            old_idx = row['original_index']
            if old_idx in existing_facility_types:
                facility_type_mapping[new_idx] = existing_facility_types[old_idx]
            else:
                facility_type_mapping[new_idx] = 'truck_stop'  # Default fallback
    
    print(f"✓ Facility type mapping complete: {len(facility_type_mapping)} existing facilities mapped")
    
    type_counts = {}
    for ftype in facility_type_mapping.values():
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    print(f"✓ Facility type distribution: {type_counts}")
    
    return all_facilities, facility_type_mapping, candidate_start_idx

# =============================================================================
# DATA LOADING MAIN FUNCTION
# =============================================================================

def load_data():
    """Load and prepare all data for optimization"""
    print("Loading and preparing data...")
    
    # Load candidate locations
    candidate_path = os.path.join(script_dir, "../results/composite_prioritization_scores.csv")
    try:
        candidates_df = pd.read_csv(candidate_path)
        print(f"Loaded {len(candidates_df)} candidate locations")
    except Exception as e:
        print(f"Error loading candidates: {e}")
        return None, None, None
   
    # Load existing facilities
    existing_path = os.path.join(script_dir, "../datasets/existing_fac_without_weigh_station.shp")
    try:
        existing_facilities = gpd.read_file(existing_path)
        print(f"Loaded {len(existing_facilities)} existing facilities from shapefile")
    except:
        try:
            csv_path = existing_path.replace('.shp', '.csv')
            existing_facilities = pd.read_csv(csv_path)
            if 'Longitude' in existing_facilities.columns and 'Latitude' in existing_facilities.columns:
                geometry = [Point(lon, lat) for lon, lat in zip(existing_facilities['Longitude'], existing_facilities['Latitude'])]
                existing_facilities = gpd.GeoDataFrame(existing_facilities, geometry=geometry, crs="EPSG:4326")
                print(f"Loaded {len(existing_facilities)} existing facilities from CSV")
            else:
                print("Error: CSV file missing coordinate columns")
                return None, None, None
        except Exception as e:
            print(f"Error loading existing facilities: {e}")
            return None, None, None
    
    # Convert candidates to GeoDataFrame
    if 'geometry' not in candidates_df.columns:
        if 'centroid_x' in candidates_df.columns and 'centroid_y' in candidates_df.columns:
            x_range = (candidates_df['centroid_x'].min(), candidates_df['centroid_x'].max())
            
            if abs(x_range[0]) > 1000:  # Likely projected coordinates
                print("Converting candidates from projected to geographic coordinates")
                geometry = [Point(x, y) for x, y in zip(candidates_df['centroid_x'], candidates_df['centroid_y'])]
                candidates_gdf = gpd.GeoDataFrame(candidates_df, geometry=geometry, crs="EPSG:3857")
                candidates_gdf = candidates_gdf.to_crs("EPSG:4326")
            else:  # Already geographic
                print("Using geographic coordinates for candidates")
                geometry = [Point(x, y) for x, y in zip(candidates_df['centroid_x'], candidates_df['centroid_y'])]
                candidates_gdf = gpd.GeoDataFrame(candidates_df, geometry=geometry, crs="EPSG:4326")
        else:
            print("Error: Missing coordinate columns in candidates")
            return None, None, None
    else:
        candidates_gdf = gpd.GeoDataFrame(candidates_df, geometry='geometry', crs="EPSG:4326")
    
    # Ensure consistent CRS
    if existing_facilities.crs != candidates_gdf.crs:
        print(f"Converting existing facilities CRS from {existing_facilities.crs} to {candidates_gdf.crs}")
        existing_facilities = existing_facilities.to_crs(candidates_gdf.crs)
    
    # Filter out problematic candidates
    initial_count = len(candidates_gdf)
    
    # Remove facilities with missing or invalid names
    candidates_gdf = candidates_gdf[
        candidates_gdf['ComplexNam'].notna() & 
        (candidates_gdf['ComplexNam'] != '') &
        (candidates_gdf['ComplexNam'] != '0')
    ]
    name_filtered = initial_count - len(candidates_gdf)
    print(f"Filtered out {name_filtered} facilities with invalid names")
    
    # Remove inappropriate facility types
    for excluded_type in EXCLUDED_FACILITY_TYPES:
        candidates_gdf = candidates_gdf[~candidates_gdf['ComplexNam'].str.contains(excluded_type, case=False, na=False)]
    
    type_filtered = initial_count - name_filtered - len(candidates_gdf)
    print(f"Filtered out {type_filtered} inappropriate facility types")
    print(f"Final candidate count: {len(candidates_gdf)}")
    
    # Load REAL traffic segments
    traffic_segments = load_real_traffic_segments()
    if traffic_segments is None:
        print("Error: Failed to load traffic segments")
        return None, None, None
    
    # Load county boundary data (optional)
    counties_gdf = load_county_data()
    
    # Add location information to candidates
    candidates_gdf = add_location_info_to_candidates(candidates_gdf, counties_gdf)
    
    # Calculate development costs
    candidates_gdf = calculate_development_costs(candidates_gdf)
    
    return candidates_gdf, existing_facilities, traffic_segments

# =============================================================================
# OPTIMIZATION ENGINE - CORRECTED VERSION
# =============================================================================

def facility_choice_optimization(candidate_idx, all_facilities, candidates_gdf, traffic_segments, 
                                lp_calculator, current_facilities, current_facility_types, 
                                candidate_start_idx, budget_remaining, proximity_analyzer):
    """Optimized facility choice with real demand analysis and proximity intelligence"""
    
    # Get candidate information
    candidate_original_idx = candidate_idx - candidate_start_idx
    if candidate_original_idx < 0 or candidate_original_idx >= len(candidates_gdf):
        return None
    
    candidate_row = candidates_gdf.iloc[candidate_original_idx]
    candidate_lat = candidate_row.geometry.y
    candidate_lon = candidate_row.geometry.x
    
    # Check budget constraints first
    if candidate_row['cost_no_service'] > budget_remaining and candidate_row['cost_full_service'] > budget_remaining:
        return None  # Neither option is affordable
    
    # Step 1: Analyze local demand patterns using REAL traffic data
    try:
        demand_analysis = analyze_demand_patterns(candidate_lat, candidate_lon, traffic_segments)
    except Exception as e:
        print(f"Error in demand analysis: {e}")
        return None
    
    # Step 2: Analyze local infrastructure
    try:
        proximity_analysis = proximity_analyzer.analyze_local_infrastructure(
            candidate_lat, candidate_lon, current_facility_types
        )
    except Exception as e:
        print(f"Error in proximity analysis: {e}")
        return None
    
    # Step 3: Integrate analyses with weighted decision making
    if demand_analysis['recommended_type'] == proximity_analysis['recommended_type']:
        selected_facility_type = demand_analysis['recommended_type']
        decision_confidence = min(0.95, (demand_analysis['confidence'] + proximity_analysis['confidence']) / 2 + 0.2)
        decision_reasoning = f"Both analyses agree: {demand_analysis['recommended_type']}"
    else:
        # Use weighted decision - prioritize demand analysis for real data-driven decisions
        demand_weight = 0.7  # Increased weight for demand analysis
        proximity_weight = 0.3
        
        if (demand_analysis['confidence'] * demand_weight) > (proximity_analysis['confidence'] * proximity_weight):
            selected_facility_type = demand_analysis['recommended_type']
            decision_confidence = demand_analysis['confidence'] * 0.8
            decision_reasoning = f"Demand analysis preferred: {demand_analysis['recommended_type']}"
        else:
            selected_facility_type = proximity_analysis['recommended_type']
            decision_confidence = proximity_analysis['confidence'] * 0.8
            decision_reasoning = f"Proximity analysis preferred: {proximity_analysis['recommended_type']}"
    
    # Step 4: Calculate costs and capacity based on selected type
    if selected_facility_type == 'rest_area':
        cost = candidate_row['cost_no_service']
        effective_capacity = candidate_row['capped_capacity_no_service']
        service_level = 'No Service'
    else:
        cost = candidate_row['cost_full_service'] 
        effective_capacity = candidate_row['capped_capacity_full_service']
        service_level = 'Full Service'
    
    # Final budget check
    if cost > budget_remaining:
        return None
    
    # Step 5: Calculate performance using REAL unmet demand
    try:
        # Get current unmet demand
        current_results = lp_calculator.calculate_unmet_demand_lp(current_facilities, current_facility_types)
        current_unmet = current_results.get('unmet_total', 0)
        
        # Test with this facility
        original_capacity = all_facilities.loc[candidate_idx, 'capacity']
        all_facilities.loc[candidate_idx, 'capacity'] = effective_capacity
        
        test_facilities = current_facilities + [candidate_idx]
        test_facility_types = current_facility_types.copy()
        test_facility_types[candidate_idx] = selected_facility_type
        
        test_results = lp_calculator.calculate_unmet_demand_lp(test_facilities, test_facility_types)
        test_unmet = test_results.get('unmet_total', 0)
        
        # Restore original capacity
        all_facilities.loc[candidate_idx, 'capacity'] = original_capacity
        
        # Calculate metrics
        demand_reduction = max(0, current_unmet - test_unmet)
        cost_effectiveness = demand_reduction / cost if cost > 0 else 0
        
        # Calculate class-specific reductions
        class_reductions = {}
        for class_id in range(1, 5):
            current_class = current_results.get(f'unmet_class_{class_id}', 0)
            test_class = test_results.get(f'unmet_class_{class_id}', 0)
            class_reductions[f'class_{class_id}_reduction'] = max(0, current_class - test_class)
        
        # Calculate composite score with facility type appropriateness
        secondary_factors = [
            candidate_row.get('crash_risk_norm', 0.5),
            candidate_row.get('accessibility_norm', 0.5),
            candidate_row.get('traffic_influx_norm', 0.5),
            candidate_row.get('capacity_norm', 0.5)
        ]
        secondary_score = sum(secondary_factors) / len(secondary_factors)
        
        # Facility-type-specific demand scoring
        if selected_facility_type == 'rest_area':
            relevant_reduction = class_reductions.get('class_1_reduction', 0) + class_reductions.get('class_3_reduction', 0)
            relevant_total_unmet = current_results.get('unmet_class_1', 0) + current_results.get('unmet_class_3', 0)
        else:  # truck_stop
            relevant_reduction = class_reductions.get('class_2_reduction', 0) + class_reductions.get('class_4_reduction', 0)
            relevant_total_unmet = current_results.get('unmet_class_2', 0) + current_results.get('unmet_class_4', 0)
        
        # Calculate facility-appropriate demand score
        if relevant_total_unmet > 0:
            facility_appropriate_score = min(1.0, relevant_reduction / (relevant_total_unmet * 0.03))
        else:
            facility_appropriate_score = 0
        
        # Overall demand score
        if current_unmet > 0:
            overall_demand_score = min(1.0, demand_reduction / (current_unmet * 0.05))
        else:
            overall_demand_score = 0
        
        # Combine scores
        demand_score = 0.7 * facility_appropriate_score + 0.3 * overall_demand_score
        
        # Confidence boost
        confidence_boost = (decision_confidence - 0.5) * 0.15
        
        # Enhanced composite score
        composite_score = PRIMARY_WEIGHT * demand_score + SECONDARY_WEIGHT * secondary_score + confidence_boost
        
     
        
        # Return configuration
        return {
            'facility_type': selected_facility_type,
            'service_level': service_level,
            'cost': cost,
            'effective_capacity': effective_capacity,
            'demand_reduction': demand_reduction,
            'composite_score': composite_score,
            'cost_effectiveness': cost_effectiveness,
            'decision_confidence': decision_confidence,
            'decision_reasoning': decision_reasoning,
            'demand_analysis': demand_analysis,
            'proximity_analysis': proximity_analysis,
            'class_reductions': class_reductions
        }
        
    except Exception as e:
        print(f"Error calculating performance: {e}")
        return None

def unified_budget_optimization(candidates_gdf, existing_facilities, traffic_segments, max_budget):
    """Main optimization function using real traffic data and proper FHWA methodology"""
    print(f"\n{'='*80}")
    print(f"CORRECTED UNIFIED OPTIMIZATION: ${max_budget/1e6:.0f}M Budget")
    print('='*80)
    
    # Prepare facilities for optimization
    try:
        all_facilities, existing_facility_types, candidate_start_idx = prepare_facilities_for_optimization(
            candidates_gdf, existing_facilities, traffic_segments
        )
        print(f"✓ Prepared {len(all_facilities)} total facilities ({candidate_start_idx} existing, {len(all_facilities) - candidate_start_idx} candidates)")
    except Exception as e:
        print(f"ERROR in facility preparation: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Initialize calculators
    try:
        lp_calculator = UnmetDemandCalculator(traffic_segments, all_facilities)
        proximity_analyzer = ProximityAnalyzer(all_facilities)
        print(f"✓ Calculators initialized with real traffic data")
        
        # Test baseline calculation
        baseline_results = lp_calculator.calculate_unmet_demand_lp([], {})
        baseline_unmet = baseline_results.get('unmet_total', 0)
        print(f"✓ Baseline unmet demand: {baseline_unmet:.1f} trucks")
        
        if baseline_unmet <= 0:
            print(f"WARNING: No unmet demand found in baseline calculation")
        
    except Exception as e:
        print(f"ERROR in calculator initialization: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Initialize optimization state
    current_facilities = list(range(candidate_start_idx))
    current_facility_types = existing_facility_types.copy()
    remaining_candidates = list(range(candidate_start_idx, len(all_facilities)))
    
    print(f"✓ Optimization state initialized:")
    print(f"  Current facilities (existing): {len(current_facilities)}")
    print(f"  Current facility types: {len(current_facility_types)} mapped")
    print(f"  Remaining candidates: {len(remaining_candidates)}")
    
    # Calculate initial state
    try:
        initial_results = lp_calculator.calculate_unmet_demand_lp(current_facilities, current_facility_types)
        initial_unmet = initial_results.get('unmet_total', 0)
        
        print(f"✓ Initial unmet demand with existing facilities: {initial_unmet:.1f} trucks")
        print(f"✓ Existing facilities reduce demand by: {baseline_unmet - initial_unmet:.1f} trucks")
        
        # Debug: Show class-wise unmet demand
        for class_id in range(1, 5):
            class_unmet = initial_results.get(f'unmet_class_{class_id}', 0)
            print(f"  Class {class_id} unmet: {class_unmet:.1f} trucks")
        
    except Exception as e:
        print(f"ERROR calculating initial state: {e}")
        import traceback
        traceback.print_exc()
        initial_unmet = baseline_unmet
    
    # Track optimization progress
    selections = []
    cumulative_budget = 0
    cumulative_capacity = 0
    selection_count = 0
    
    # Track budget progression
    budget_progression = [{
        'budget_used': 0,
        'cumulative_capacity': 0,
        'facilities_count': len(current_facilities),
        'unmet_class_1': initial_results.get('unmet_class_1', 0),
        'unmet_class_2': initial_results.get('unmet_class_2', 0),
        'unmet_class_3': initial_results.get('unmet_class_3', 0),
        'unmet_class_4': initial_results.get('unmet_class_4', 0),
        'unmet_total': initial_unmet,
        'facility_added': 'baseline'
    }]
    
    print(f"\nStarting optimization with {len(remaining_candidates)} candidates...")
    
    # Main optimization loop
    max_iterations = min(50, len(remaining_candidates))
    
    for iteration in range(max_iterations):
        if not remaining_candidates or cumulative_budget >= max_budget:
            break
        
        budget_remaining = max_budget - cumulative_budget
        print(f"\nIteration {iteration + 1}: Budget remaining ${budget_remaining/1e6:.2f}M, Candidates: {len(remaining_candidates)}")
        
        # Evaluate candidates
        candidate_evaluations = []
        
        for candidate_idx in remaining_candidates:
            try:
                config = facility_choice_optimization(
                    candidate_idx, all_facilities, candidates_gdf, traffic_segments,
                    lp_calculator, current_facilities, current_facility_types,
                    candidate_start_idx, budget_remaining, proximity_analyzer
                )
                
                if config is not None:
                    candidate_evaluations.append({
                        'candidate_idx': candidate_idx,
                        'config': config
                    })
            except:
                continue
        
        if not candidate_evaluations:
            print(f"No valid candidates found - stopping optimization")
            break
        
        # Select the best candidate
        candidate_evaluations.sort(key=lambda x: x['config']['composite_score'], reverse=True)
        best_candidate = candidate_evaluations[0]
        
        selected_idx = best_candidate['candidate_idx']
        best_config = best_candidate['config']
        
        # Minimum score threshold
        if best_config['composite_score'] < 0.001:
            print(f"Best score {best_config['composite_score']:.6f} below threshold - stopping")
            break
        
        print(f"  Selected facility {selected_idx}: {best_config['facility_type']} - Score: {best_config['composite_score']:.4f}")
        
        # Add the selected facility
        selection_count += 1
        cumulative_budget += best_config['cost']
        
        # Get facility details
        candidate_original_idx = selected_idx - candidate_start_idx
        candidate_row = candidates_gdf.iloc[candidate_original_idx]
        current_capacity = best_config['effective_capacity']
        cumulative_capacity += current_capacity
        
        # Update optimization state
        current_facilities.append(selected_idx)
        current_facility_types[selected_idx] = best_config['facility_type']
        remaining_candidates.remove(selected_idx)
        
        # Calculate updated unmet demand
        try:
            updated_results = lp_calculator.calculate_unmet_demand_lp(current_facilities, current_facility_types)
            updated_unmet = updated_results.get('unmet_total', 0)
            
            # Track budget progression
            budget_progression.append({
                'budget_used': cumulative_budget,
                'cumulative_capacity': cumulative_capacity,
                'facilities_count': len(current_facilities),
                'unmet_class_1': updated_results.get('unmet_class_1', 0),
                'unmet_class_2': updated_results.get('unmet_class_2', 0),
                'unmet_class_3': updated_results.get('unmet_class_3', 0),
                'unmet_class_4': updated_results.get('unmet_class_4', 0),
                'unmet_total': updated_unmet,
                'facility_added': candidate_row.get('ComplexNam', f'Facility_{selected_idx}')
            })
            
            print(f"    Updated unmet demand: {updated_unmet:.1f} trucks (reduction: {initial_unmet - updated_unmet:.1f})")
            
        except Exception as e:
            print(f"    Error calculating updated unmet demand: {e}")
        
        # Get county information
        facility_county = candidate_row.get('county', 'Unknown')
        if facility_county == 'Unknown':
            lat, lon = candidate_row.geometry.y, candidate_row.geometry.x
            nearby_segments = traffic_segments[
                (abs(traffic_segments['Latitude'] - lat) < 0.1) & 
                (abs(traffic_segments['Longitude'] - lon) < 0.1)
            ]
            if len(nearby_segments) > 0:
                facility_county = nearby_segments.iloc[0]['assigned_county']
        
        # Record selection
        selection_record = {
            'Selection_Order': selection_count,
            'FID': candidate_row.get('FID', selected_idx),
            'Facility_Name': candidate_row.get('ComplexNam', f'Facility_{selected_idx}'),
            'County': facility_county,
            'City': candidate_row.get('city', 'Unknown'),
            'Service_Level': best_config['service_level'],
            'Facility_Type': best_config['facility_type'],
            'Cost': best_config['cost'],
            'Original_Capacity': candidate_row['capacity_value'],
            'Effective_Capacity': best_config['effective_capacity'],
            'Cumulative_Budget': cumulative_budget,
            'Cumulative_Capacity': cumulative_capacity,
            'Composite_Score': best_config['composite_score'],
            'Demand_Reduction': best_config['demand_reduction'],
            'Cost_Effectiveness': best_config['cost_effectiveness'],
            'Decision_Confidence': best_config['decision_confidence'],
            'Decision_Reasoning': best_config['decision_reasoning'],
            'centroid_x': candidate_row.geometry.x,
            'centroid_y': candidate_row.geometry.y
        }
        
        # Add demand class reductions
        if 'class_reductions' in best_config:
            for class_key, reduction_value in best_config['class_reductions'].items():
                selection_record[class_key] = reduction_value
        
        # Add current unmet demand levels
        try:
            selection_record['current_unmet_class_1'] = updated_results.get('unmet_class_1', 0)
            selection_record['current_unmet_class_2'] = updated_results.get('unmet_class_2', 0)
            selection_record['current_unmet_class_3'] = updated_results.get('unmet_class_3', 0)
            selection_record['current_unmet_class_4'] = updated_results.get('unmet_class_4', 0)
            selection_record['current_unmet_total'] = updated_results.get('unmet_total', 0)
        except:
            pass
        
        selections.append(selection_record)
    
    # Create results DataFrames
    results_df = pd.DataFrame(selections)
    budget_progression_df = pd.DataFrame(budget_progression)
    
    print(f"\nOptimization Complete:")
    print(f"  Facilities selected: {len(results_df)}")
    if len(results_df) > 0:
        print(f"  Budget used: ${cumulative_budget/1e6:.2f}M ({cumulative_budget/max_budget*100:.1f}%)")
        print(f"  Capacity added: {cumulative_capacity:,}")
        print(f"  Average score: {results_df['Composite_Score'].mean():.4f}")
        
        # Show type distribution
        if 'Facility_Type' in results_df.columns:
            type_dist = results_df['Facility_Type'].value_counts().to_dict()
            print(f"  Type distribution: {type_dist}")
        
        # Show county distribution
        if 'County' in results_df.columns:
            county_dist = results_df['County'].value_counts().head(5).to_dict()
            print(f"  Top counties: {county_dist}")
    
    return results_df, budget_progression_df
    

# =============================================================================
# VISUALIZATION AND REPORTING
# =============================================================================

def create_optimization_plots(results, budget_scenario, output_folder):
    """Create visualization plots for optimization results"""
    
    if len(results) == 0:
        print("No results to plot")
        return
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Corrected Optimization Results - ${budget_scenario/1e6:.0f}M Budget', fontsize=16, fontweight='bold')
        
        # Plot 1: Budget utilization
        ax1.step(results['Cumulative_Budget'] / 1e6, results['Selection_Order'], 
                where='post', linewidth=2.5, color='blue', marker='o', markersize=4)
        ax1.set_xlabel('Budget ($ Millions)')
        ax1.set_ylabel('Number of Facilities')
        ax1.set_title('Facilities Selected vs Budget', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Capacity growth
        ax2.step(results['Cumulative_Budget'] / 1e6, results['Cumulative_Capacity'], 
                where='post', linewidth=2.5, color='green', marker='s', markersize=4)
        ax2.set_xlabel('Budget ($ Millions)')
        ax2.set_ylabel('Cumulative Capacity (Spaces)')
        ax2.set_title('Capacity Added vs Budget', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Facility type distribution
        if 'Facility_Type' in results.columns:
            type_counts = results['Facility_Type'].value_counts()
            colors = ['lightcoral', 'lightblue']
            wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax3.set_title('Facility Type Distribution', fontweight='bold')
        
        # Plot 4: Performance metrics
        ax4.plot(results['Selection_Order'], results['Composite_Score'], 
                'o-', linewidth=2, markersize=6, color='red')
        ax4.set_xlabel('Selection Order')
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Composite Score Evolution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cumulative Unmet Demand Reduction by Class vs Budget
        ax5.set_title('Cumulative Unmet Demand Reduction by Class vs Budget', fontweight='bold')
        
        class_columns = [col for col in results.columns if col.startswith('class_') and col.endswith('_reduction')]
        
        if class_columns:
            class_styles = {
                'class_1_reduction': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Class 1 (SH Rest Area)'},
                'class_2_reduction': {'color': '#ff7f0e', 'linestyle': '--', 'label': 'Class 2 (SH Truck Stop)'},
                'class_3_reduction': {'color': '#2ca02c', 'linestyle': '-.', 'label': 'Class 3 (LH Rest Area)'},
                'class_4_reduction': {'color': '#d62728', 'linestyle': ':', 'label': 'Class 4 (LH Truck Stop)'}
            }
            
            for class_col in class_columns:
                if class_col in class_styles and class_col in results.columns:
                    cumulative_reduction = results[class_col].cumsum()
                    
                    ax5.step(results['Cumulative_Budget'] / 1e6, cumulative_reduction,
                            where='post', linewidth=2.5,
                            color=class_styles[class_col]['color'],
                            linestyle=class_styles[class_col]['linestyle'],
                            label=class_styles[class_col]['label'],
                            marker='o', markersize=3)
            
            ax5.set_xlabel('Budget ($ Millions)')
            ax5.set_ylabel('Cumulative Unmet Demand Reduction (trucks)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No class reduction data available', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        
        # Plot 6: Individual Facility Demand Reduction by Class
        ax6.set_title('Individual Facility Demand Reduction by Class', fontweight='bold')
        
        if class_columns:
            x = range(len(results))
            width = 0.8
            
            # Prepare data for stacking
            class_1_values = results.get('class_1_reduction', [0] * len(results)).fillna(0)
            class_2_values = results.get('class_2_reduction', [0] * len(results)).fillna(0)
            class_3_values = results.get('class_3_reduction', [0] * len(results)).fillna(0)
            class_4_values = results.get('class_4_reduction', [0] * len(results)).fillna(0)
            
            # Create stacked bars
            p1 = ax6.bar(x, class_1_values, width, label='Class 1 (SH Rest Area)', 
                        color=class_styles['class_1_reduction']['color'], alpha=0.8)
            p2 = ax6.bar(x, class_2_values, width, bottom=class_1_values, 
                        label='Class 2 (SH Truck Stop)', 
                        color=class_styles['class_2_reduction']['color'], alpha=0.8)
            
            bottom_3 = class_1_values + class_2_values
            p3 = ax6.bar(x, class_3_values, width, bottom=bottom_3, 
                        label='Class 3 (LH Rest Area)', 
                        color=class_styles['class_3_reduction']['color'], alpha=0.8)
            
            bottom_4 = bottom_3 + class_3_values
            p4 = ax6.bar(x, class_4_values, width, bottom=bottom_4, 
                        label='Class 4 (LH Truck Stop)', 
                        color=class_styles['class_4_reduction']['color'], alpha=0.8)
            
            ax6.set_xlabel('Facility Selection Order')
            ax6.set_ylabel('Demand Reduction (trucks)')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add facility names as x-tick labels
            if len(results) <= 20:
                facility_names = [name[:15] + '...' if len(name) > 15 else name 
                                for name in results['Facility_Name']]
                ax6.set_xticks(x)
                ax6.set_xticklabels(facility_names, rotation=45, ha='right')
            else:
                ax6.set_xticks(x[::max(1, len(x)//10)])
                ax6.set_xticklabels([f'F{i+1}' for i in x[::max(1, len(x)//10)]])
        else:
            ax6.text(0.5, 0.5, 'No class reduction data available', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_folder, f"corrected_optimization_{budget_scenario/1e6:.0f}M.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Optimization plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()

def create_unmet_demand_vs_budget_plots(budget_progression, scenario_name, budget_scenario, output_folder):
    """Create unmet demand vs budget plots for each class"""
    
    if len(budget_progression) == 0:
        print("No budget progression data to plot")
        return
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Unmet Demand vs Budget Progression - {scenario_name.title()} Scenario', fontsize=16, fontweight='bold')
        
        # Class-specific plots
        class_info = {
            1: {'name': 'Class 1 (Short-Haul Rest Area)', 'color': '#1f77b4'},
            2: {'name': 'Class 2 (Short-Haul Truck Stop)', 'color': '#ff7f0e'},
            3: {'name': 'Class 3 (Long-Haul Rest Area)', 'color': '#2ca02c'},
            4: {'name': 'Class 4 (Long-Haul Truck Stop)', 'color': '#d62728'}
        }
        
        axes = [ax1, ax2, ax3, ax4]
        
        for class_id, ax in zip([1, 2, 3, 4], axes):
            class_col = f'unmet_class_{class_id}'
            if class_col in budget_progression.columns:
                ax.plot(budget_progression['budget_used'] / 1e6, budget_progression[class_col],
                       'o-', linewidth=2.5, markersize=4, 
                       color=class_info[class_id]['color'])
                ax.set_title(class_info[class_id]['name'], fontweight='bold')
                ax.set_xlabel('Budget Used ($ Millions)')
                ax.set_ylabel('Unmet Demand (trucks)')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {class_info[class_id]["name"]}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_folder, f"unmet_demand_vs_budget_{scenario_name}_{budget_scenario/1e6:.0f}M.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Unmet demand vs budget plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"Error creating unmet demand vs budget plots: {e}")

def create_summary_report(results, budget_scenario, output_folder):
    """Create a text summary report of the optimization results"""
    
    try:
        report_file = os.path.join(output_folder, f"corrected_optimization_summary_{budget_scenario/1e6:.0f}M.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CORRECTED OPTIMIZATION SUMMARY REPORT - ${budget_scenario/1e6:.0f}M BUDGET\n")
            f.write("="*80 + "\n\n")
            
            f.write("METHODOLOGY IMPROVEMENTS:\n")
            f.write("  ✓ Real traffic data from GPKG files\n")
            f.write("  ✓ FHWA MILP demand calculation methodology\n")
            f.write("  ✓ Removed all random data generation\n")
            f.write("  ✓ County assignment via RouteID mapping\n")
            f.write("  ✓ Interstate assignment from data fields\n")
            f.write("  ✓ Class-specific demand calculation (1-4)\n")
            f.write("  ✓ Enhanced facility type assignment\n\n")
            
            if len(results) > 0:
                f.write(f"OVERALL RESULTS:\n")
                f.write(f"  Facilities Selected: {len(results)}\n")
                f.write(f"  Budget Used: ${results['Cumulative_Budget'].iloc[-1]/1e6:.2f}M ({results['Cumulative_Budget'].iloc[-1]/budget_scenario*100:.1f}%)\n")
                f.write(f"  Total Capacity Added: {results['Cumulative_Capacity'].iloc[-1]:,} spaces\n")
                f.write(f"  Average Composite Score: {results['Composite_Score'].mean():.4f}\n")
                
                if 'Decision_Confidence' in results.columns:
                    f.write(f"  Average Decision Confidence: {results['Decision_Confidence'].mean():.3f}\n")
                
                f.write(f"\nFACILITY TYPE DISTRIBUTION:\n")
                if 'Facility_Type' in results.columns:
                    type_dist = results['Facility_Type'].value_counts()
                    for ftype, count in type_dist.items():
                        f.write(f"  {ftype.replace('_', ' ').title()}: {count} facilities\n")
                
                f.write(f"\nSERVICE LEVEL DISTRIBUTION:\n")
                if 'Service_Level' in results.columns:
                    service_dist = results['Service_Level'].value_counts()
                    for service, count in service_dist.items():
                        f.write(f"  {service}: {count} facilities\n")
                
                f.write(f"\nGEOGRAPHIC DISTRIBUTION:\n")
                if 'County' in results.columns:
                    county_dist = results['County'].value_counts().head(10)
                    for county, count in county_dist.items():
                        f.write(f"  {county}: {count} facilities\n")
                
                f.write(f"\nDEMAND CLASS IMPACT:\n")
                class_columns = [col for col in results.columns if col.startswith('class_') and col.endswith('_reduction')]
                class_descriptions = get_demand_class_descriptions()
                
                for class_col in class_columns:
                    if class_col in results.columns:
                        class_num = class_col.split('_')[1]
                        class_desc = class_descriptions.get(f'class_{class_num}', f'Class {class_num}')
                        total_reduction = results[class_col].sum()
                        facilities_serving = (results[class_col] > 0).sum()
                        f.write(f"  {class_desc}: {total_reduction:.1f} total reduction, {facilities_serving} facilities\n")
                
                f.write(f"\nTOP 10 SELECTED FACILITIES:\n")
                for i in range(min(10, len(results))):
                    row = results.iloc[i]
                    f.write(f"  {i+1:2d}. {row['Facility_Name'][:30]:30s} | {row['County'][:15]:15s} | {row['Facility_Type']:10s} | ${row['Cost']/1e6:5.2f}M | Score: {row['Composite_Score']:.4f}\n")
                
            else:
                f.write("NO FACILITIES SELECTED\n")
                f.write("Check optimization parameters and data quality\n")
        
        print(f"Summary report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error creating summary report: {e}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main execution function with corrected methodology"""
    print("=" * 80)
    print("CORRECTED UNIFIED FACILITY OPTIMIZATION SYSTEM")
    print("Using Real Traffic Data and FHWA Methodology")
    print("=" * 80)
    
    try:
        # Load real data
        print(f"\n{'='*60}")
        print("DATA LOADING WITH REAL SOURCES")
        print('='*60)
        
        candidates_gdf, existing_facilities, traffic_segments = load_data()
        
        if candidates_gdf is None or existing_facilities is None or traffic_segments is None:
            print("ERROR: Failed to load required data. Exiting.")
            return
        
        print(f"\n✓ Data loaded successfully:")
        print(f"  Candidates: {len(candidates_gdf)}")
        print(f"  Existing facilities: {len(existing_facilities)}")
        print(f"  Traffic segments: {len(traffic_segments)}")
        print(f"  Total demand: {traffic_segments['demand_total'].sum():.0f} trucks")
        
        # Create output folders
        output_folder = os.path.join(script_dir, "../results/corrected_optimization")
        figures_folder = os.path.join(output_folder, "figures")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(figures_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Output folders created:")
        print(f"  Results: {output_folder}")
        print(f"  Figures: {figures_folder}")
        
        # Run optimization for both budget scenarios
        results_dict = {}
        budget_progression_dict = {}
        
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            print(f"\n{'='*80}")
            print(f"RUNNING CORRECTED OPTIMIZATION: {scenario_name.upper()} SCENARIO (${budget/1e6:.0f}M)")
            print('='*80)
            
            # Get both results and budget progression
            scenario_results, budget_progression = unified_budget_optimization(
                candidates_gdf, existing_facilities, traffic_segments, budget
            )
            
            # Store results
            results_dict[scenario_name] = scenario_results
            budget_progression_dict[scenario_name] = budget_progression
            
            # Save CSV results
            output_file = os.path.join(output_folder, f"corrected_optimization_{scenario_name}_{budget/1e6:.0f}M.csv")
            scenario_results.to_csv(output_file, index=False)
            print(f"\n✓ Results saved: {output_file}")
            
            # Save budget progression
            budget_progression_file = os.path.join(output_folder, f"budget_progression_{scenario_name}_{budget/1e6:.0f}M.csv")
            budget_progression.to_csv(budget_progression_file, index=False)
            print(f"✓ Budget progression saved: {budget_progression_file}")
            
            # Create visualizations
            create_optimization_plots(scenario_results, budget, figures_folder)
            create_summary_report(scenario_results, budget, output_folder)
            
            # Create unmet demand vs budget plots
            create_unmet_demand_vs_budget_plots(budget_progression, scenario_name, budget, figures_folder)
            
            # Scenario summary
            print(f"\n{'='*60}")
            print(f"{scenario_name.upper()} SCENARIO SUMMARY")
            print('='*60)
            
            if len(scenario_results) > 0:
                print(f"✓ SUCCESS: Selected {len(scenario_results)} facilities")
                print(f"✓ Budget used: ${scenario_results['Cumulative_Budget'].iloc[-1]/1e6:.2f}M")
                print(f"✓ Capacity added: {scenario_results['Cumulative_Capacity'].iloc[-1]:,}")
                print(f"✓ Average score: {scenario_results['Composite_Score'].mean():.4f}")
                
                # Show type distribution
                if 'Facility_Type' in scenario_results.columns:
                    type_dist = scenario_results['Facility_Type'].value_counts()
                    print(f"✓ Types: {dict(type_dist)}")
                
                # Show county distribution
                if 'County' in scenario_results.columns:
                    county_dist = scenario_results['County'].value_counts().head(3)
                    print(f"✓ Top counties: {dict(county_dist)}")
                    
                # Show final unmet demand by class
                if 'current_unmet_class_1' in scenario_results.columns:
                    final_unmet = scenario_results.iloc[-1]
                    print(f"✓ Final unmet demand:")
                    print(f"  Class 1: {final_unmet.get('current_unmet_class_1', 0):.1f} trucks")
                    print(f"  Class 2: {final_unmet.get('current_unmet_class_2', 0):.1f} trucks") 
                    print(f"  Class 3: {final_unmet.get('current_unmet_class_3', 0):.1f} trucks")
                    print(f"  Class 4: {final_unmet.get('current_unmet_class_4', 0):.1f} trucks")
                    print(f"  Total: {final_unmet.get('current_unmet_total', 0):.1f} trucks")
                    
                # Show top 3 facilities
                print(f"\nTop 3 Selected Facilities:")
                for i in range(min(3, len(scenario_results))):
                    row = scenario_results.iloc[i]
                    print(f"  {i+1}. {row['Facility_Name'][:25]:25s} | {row['County']:15s} | {row['Facility_Type']:10s} | ${row['Cost']/1e6:5.2f}M")
                
            else:
                print("✗ No facilities selected - check data and parameters")
        
        # Final comparative summary
        print(f"\n{'='*80}")
        print("FINAL COMPARATIVE SUMMARY")
        print('='*80)
        
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            results = results_dict[scenario_name]
            print(f"\n{scenario_name.upper()} SCENARIO (${budget/1e6:.0f}M):")
            
            if len(results) > 0:
                print(f"  ✓ Facilities: {len(results)}")
                print(f"  ✓ Budget used: ${results['Cumulative_Budget'].iloc[-1]/1e6:.2f}M")
                print(f"  ✓ Capacity: {results['Cumulative_Capacity'].iloc[-1]:,}")
                print(f"  ✓ Avg score: {results['Composite_Score'].mean():.4f}")
            else:
                print(f"  ✗ No facilities selected")
        
        print(f"\n✓ All outputs saved to: {output_folder}")
        print(f"✓ Methodology corrected: Real data, FHWA demand calculation, no random assumptions")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
