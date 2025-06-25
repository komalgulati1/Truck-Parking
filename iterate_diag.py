#!/usr/bin/env python3
"""
DIAGNOSTIC FACILITY OPTIMIZATION SYSTEM
Complete diagnostic script to identify and fix optimization issues
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pulp
import warnings
import requests
import time
from geopy.geocoders import Nominatim
warnings.filterwarnings('ignore')

# Define budget scenarios for comprehensive analysis
BUDGET_SCENARIOS = {
    'current': 175e6,      # Current budget constraint (175 million)
    'expanded': 1000e6     # Expanded budget scenario (1 billion)
}

# Define excluded facility types
EXCLUDED_FACILITY_TYPES = ["DMV", "License", "Welcome Center", "Municipal", "Courthouse", "Rest Area"]

# Weights for demand-based composite scoring
PRIMARY_WEIGHT = 0.5  # Unmet demand reduction
SECONDARY_WEIGHT = 0.5  # Traditional factors (crash, accessibility, etc.)

# Proximity analysis parameters for intelligent facility type selection
PROXIMITY_ANALYSIS_RADIUS = 25.0  # Miles - radius for analyzing local infrastructure
TRUCK_STOP_SPACING_THRESHOLD = 15.0  # Miles - minimum spacing between truck stops
REST_AREA_SPACING_THRESHOLD = 10.0   # Miles - minimum spacing between rest areas

# File paths - using actual data sources
TRAFFIC_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/results/traffic_segments_interstates_only.gpkg'
INTERSTATE_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/interstate_ncdot.gpkg'
EXISTING_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/existing.gpkg'
CANDIDATE_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/shortlisted_candidates.shp'
COUNTY_PATH = '/Users/komalgulati/Documents/Project_3_1/simulation/datasets/county.csv'
OUTPUT_DIR = '/Users/komalgulati/Documents/Project_3_1/simulation/results/unified_optimization'

# Get the current directory of the script
script_dir = os.path.dirname(__file__)
if not script_dir:
    script_dir = '.'

def get_demand_class_descriptions():
    """Return descriptions of demand classes for reporting"""
    return {
        'class_1': 'Short-Haul Rest Area (basic service, day trips)',
        'class_2': 'Short-Haul Truck Stop (full service, day trips)', 
        'class_3': 'Long-Haul Rest Area (basic service, overnight)',
        'class_4': 'Long-Haul Truck Stop (full service, overnight)'
    }

class UnmetDemandCalculator:
    """
    Calculate unmet demand using Linear Programming with variable distance thresholds
    """
    
    def __init__(self, traffic_segments, facilities, base_distance_threshold=50.0):
        self.traffic_segments = traffic_segments.copy()
        self.facilities = facilities.copy()
        self.base_distance_threshold = base_distance_threshold
        
        print(f"Initializing demand calculator with {len(traffic_segments)} segments and {len(facilities)} facilities")
        
        # Validate coordinates
        self._validate_coordinates()
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Calculate segment-specific thresholds
        self.segment_thresholds = self._calculate_segment_thresholds()
        
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
    
    def _calculate_segment_thresholds(self):
        """Calculate variable distance thresholds for each segment"""
        segment_thresholds = {}
        
        existing_facilities = [
            j for j in range(len(self.facilities)) 
            if self.facilities.iloc[j].get('facility_type', '') == 'existing'
        ]
        
        print(f"Calculating thresholds based on {len(existing_facilities)} existing facilities")
        
        for i in range(len(self.traffic_segments)):
            if existing_facilities:
                existing_dists = [self.distance_matrix[i, j] for j in existing_facilities 
                                if not np.isnan(self.distance_matrix[i, j])]
                
                if len(existing_dists) >= 4:
                    threshold = sorted(existing_dists)[3]  # 4th closest
                elif existing_dists:
                    threshold = max(existing_dists)
                else:
                    threshold = self.base_distance_threshold
            else:
                threshold = self.base_distance_threshold
            
            segment_thresholds[i] = threshold
        
        return segment_thresholds
    
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
                    radius = self.segment_thresholds.get(i, self.base_distance_threshold)
                    
                    for j in compatible_facilities:
                        if (not np.isnan(self.distance_matrix[i, j]) and 
                            self.distance_matrix[i, j] <= radius):
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

class ProximityAnalyzer:
    """
    Intelligent proximity analysis for facility type selection
    """
    
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
        """
        Analyze local infrastructure around a candidate location to determine optimal facility type
        """
        
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
        
        # Strategic facility type recommendation based on local infrastructure gaps
        recommendation = self._determine_facility_type_recommendation(
            distances_and_types, nearby_truck_stops, nearby_rest_areas
        )
        
        return recommendation
    
    def _determine_facility_type_recommendation(self, nearby_facilities, nearby_truck_stops, nearby_rest_areas):
        """
        Apply strategic logic to recommend facility type based on local infrastructure analysis
        """
        
        # If no nearby facilities, prefer truck stop (comprehensive service for underserved area)
        if not nearby_facilities:
            return {
                'recommended_type': 'truck_stop',
                'confidence': 0.8,
                'reasoning': 'No nearby facilities - truck stop provides comprehensive coverage'
            }
        
        # Check for very close truck stops (competition concern)
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
        
        # Balance assessment: what type of facility is most needed in this area?
        truck_stop_density = len([f for f in nearby_truck_stops if f['distance'] <= 20])
        rest_area_density = len([f for f in nearby_rest_areas if f['distance'] <= 20])
        
        # Enhanced logic to better balance facility types
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
        elif truck_stop_density > rest_area_density + 1:  # Significantly more truck stops
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

def load_county_data():
    """Load county boundary data for North Carolina"""
    print("Loading county boundary data...")
    
    # Option 1: Try to load from local shapefile
    county_paths = [
        os.path.join(script_dir, "../datasets/nc_counties.shp"),
        os.path.join(script_dir, "../datasets/counties.shp"),
        os.path.join(script_dir, "../datasets/NC_counties.shp")
    ]
    
    for path in county_paths:
        try:
            counties_gdf = gpd.read_file(path)
            print(f"Loaded county data from: {path}")
            # Ensure consistent CRS
            if counties_gdf.crs != "EPSG:4326":
                counties_gdf = counties_gdf.to_crs("EPSG:4326")
            return counties_gdf
        except:
            continue
    
    print("Warning: County boundary data not found. Will use reverse geocoding for location names.")
    return None

def get_location_info(lat, lon, counties_gdf=None):
    """Get county/city information for a given coordinate"""
    if counties_gdf is not None:
        # Method 1: Use county boundaries (most accurate)
        try:
            point = Point(lon, lat)
            
            # Find which county contains this point
            for idx, county in counties_gdf.iterrows():
                if county.geometry.contains(point):
                    county_name = county.get('NAME', county.get('COUNTY', county.get('name', 'Unknown')))
                    return {
                        'county': county_name,
                        'city': 'Unknown',  # Would need city boundaries for this
                        'method': 'spatial_join'
                    }
        except Exception as e:
            print(f"Error in spatial join: {e}")
    
    # Method 2: Use reverse geocoding (backup method)
    try:
        # Simple reverse geocoding using Nominatim (free but slower)
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
    
    # Method 3: Fallback - determine county based on coordinate ranges (NC specific)
    return get_nc_county_estimate(lat, lon)

def get_nc_county_estimate(lat, lon):
    """Rough county estimation for North Carolina based on coordinate ranges"""
    # Major NC counties with approximate coordinate ranges
    nc_counties = {
        'Mecklenburg': {'lat_range': (35.0, 35.5), 'lon_range': (-81.1, -80.5)},  # Charlotte area
        'Wake': {'lat_range': (35.6, 36.0), 'lon_range': (-78.9, -78.3)},        # Raleigh area
        'Guilford': {'lat_range': (35.9, 36.3), 'lon_range': (-80.1, -79.6)},    # Greensboro area
        'Forsyth': {'lat_range': (36.0, 36.3), 'lon_range': (-80.4, -80.0)},     # Winston-Salem
        'Durham': {'lat_range': (35.8, 36.1), 'lon_range': (-79.1, -78.7)},      # Durham area
        'Cumberland': {'lat_range': (34.9, 35.4), 'lon_range': (-79.2, -78.6)},  # Fayetteville
        'New Hanover': {'lat_range': (34.1, 34.4), 'lon_range': (-78.1, -77.7)}, # Wilmington
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
    
    # Batch process to avoid overwhelming reverse geocoding services
    for idx, candidate in candidates_gdf.iterrows():
        try:
            lat = candidate.geometry.y
            lon = candidate.geometry.x
            
            location_info = get_location_info(lat, lon, counties_gdf)
            
            counties.append(location_info['county'])
            cities.append(location_info['city'])
            location_methods.append(location_info['method'])
            
            # Add small delay for reverse geocoding to be respectful
            if location_info['method'] == 'reverse_geocoding':
                time.sleep(0.1)  # 100ms delay
                
        except Exception as e:
            print(f"Error processing candidate {idx}: {e}")
            counties.append('Unknown')
            cities.append('Unknown')
            location_methods.append('error')
    
    candidates_gdf['county'] = counties
    candidates_gdf['city'] = cities
    candidates_gdf['location_method'] = location_methods
    
    print(f"Location information added to {len(candidates_gdf)} candidates")
    
    # Report on location methods used
    method_counts = pd.Series(location_methods).value_counts()
    print(f"Location determination methods: {dict(method_counts)}")
    
    return candidates_gdf

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
    
    # Calculate maximum feasible capacity for each facility type
    max_capacity_no_service = (MAX_NO_SERVICE_COST - SITE_PREP_NO_SERVICE) // COST_PER_SPACE_NO_SERVICE
    max_capacity_full_service = (MAX_FULL_SERVICE_COST - SITE_PREP_FULL_SERVICE) // COST_PER_SPACE_FULL_SERVICE
    
    print(f"Capacity constraints derived from maximum costs:")
    print(f"  Basic facilities (rest areas): maximum {max_capacity_no_service} spaces")
    print(f"  Full-service facilities (truck stops): maximum {max_capacity_full_service} spaces")
    
    # Apply capacity caps
    candidates_gdf['capped_capacity_no_service'] = candidates_gdf['capacity_value'].clip(upper=max_capacity_no_service)
    candidates_gdf['capped_capacity_full_service'] = candidates_gdf['capacity_value'].clip(upper=max_capacity_full_service)
    
    # Calculate costs using the capped capacities
    candidates_gdf['cost_no_service'] = (SITE_PREP_NO_SERVICE + 
                                       candidates_gdf['capped_capacity_no_service'] * COST_PER_SPACE_NO_SERVICE)
    
    candidates_gdf['cost_full_service'] = (SITE_PREP_FULL_SERVICE + 
                                         candidates_gdf['capped_capacity_full_service'] * COST_PER_SPACE_FULL_SERVICE)
    
    # Verify that no costs exceed the maximum values
    assert candidates_gdf['cost_no_service'].max() <= MAX_NO_SERVICE_COST, "No-service costs exceed maximum!"
    assert candidates_gdf['cost_full_service'].max() <= MAX_FULL_SERVICE_COST, "Full-service costs exceed maximum!"
    
    # Report capacity adjustments for transparency
    capacity_reduced_basic = (candidates_gdf['capacity_value'] > max_capacity_no_service).sum()
    capacity_reduced_full = (candidates_gdf['capacity_value'] > max_capacity_full_service).sum()
    
    print(f"Capacity adjustments applied:")
    print(f"  {capacity_reduced_basic} sites had capacity reduced for basic facilities")
    print(f"  {capacity_reduced_full} sites had capacity reduced for full-service facilities")
    
    # Report the range of original capacities for context
    print(f"Original capacity range: {candidates_gdf['capacity_value'].min():.0f} to {candidates_gdf['capacity_value'].max():.0f} spaces")
    
    # Report final cost statistics to verify everything worked correctly
    print(f"Final cost ranges:")
    print(f"  Basic facilities: ${candidates_gdf['cost_no_service'].min()/1e6:.2f}M to ${candidates_gdf['cost_no_service'].max()/1e6:.2f}M")
    print(f"  Full-service facilities: ${candidates_gdf['cost_full_service'].min()/1e6:.2f}M to ${candidates_gdf['cost_full_service'].max()/1e6:.2f}M")
    
    return candidates_gdf

def load_traffic_segments():
    """Load realistic traffic segments for North Carolina using exact FHWA parameters"""
    print("Creating traffic segments with FHWA-derived demand parameters:")
    
    # FHWA Constants - Based on actual driver surveys
    F_S = 1.15  # Seasonal peaking factor
    
    P_CLASS = {
        'urban': {'short_haul': 0.36, 'long_haul': 0.64},
        'rural': {'short_haul': 0.07, 'long_haul': 0.93}
    }
    
    P_FACILITY = {'rest_area': 0.23, 'truck_stop': 0.77}
    
    P_PEAK = {'short_haul': 0.02, 'long_haul': 0.09}
    
    P_PARK = {'short_haul': 5/60, 'long_haul': 0.783}  # Parking duration in hours
    
    print("FHWA Parameters:")
    print(f"  Seasonal factor: {F_S}")
    print(f"  Urban split - Short: {P_CLASS['urban']['short_haul']:.2f}, Long: {P_CLASS['urban']['long_haul']:.2f}")
    print(f"  Rural split - Short: {P_CLASS['rural']['short_haul']:.2f}, Long: {P_CLASS['rural']['long_haul']:.2f}")
    print(f"  Facility preference - Rest area: {P_FACILITY['rest_area']:.2f}, Truck stop: {P_FACILITY['truck_stop']:.2f}")
    print(f"  Peak factors - Short: {P_PEAK['short_haul']:.2f}, Long: {P_PEAK['long_haul']:.2f}")
    print(f"  Parking duration - Short: {P_PARK['short_haul']:.3f}h, Long: {P_PARK['long_haul']:.3f}h")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create segments distributed across NC
    n_segments = 350
    
    # Base distribution across NC
    traffic_data = {
        'Latitude': np.random.uniform(34.0, 36.5, n_segments),
        'Longitude': np.random.uniform(-84.0, -75.5, n_segments),
    }
    
    # Generate realistic AADTT values (Annual Average Daily Truck Traffic)
    base_aadtt = np.random.exponential(500, n_segments)  # Base truck traffic
    
    # Assign urban/rural classification (simplified)
    # Major urban counties in NC
    urban_counties = ['Wake', 'Mecklenburg', 'Durham', 'Guilford', 'Forsyth', 'Cumberland', 'Buncombe', 'New Hanover']
    
    # Assign area type based on coordinates (simplified approach)
    is_urban = []
    for lat, lon in zip(traffic_data['Latitude'], traffic_data['Longitude']):
        # Charlotte area (Mecklenburg)
        if 35.0 <= lat <= 35.5 and -81.1 <= lon <= -80.5:
            is_urban.append(True)
        # Raleigh area (Wake)
        elif 35.6 <= lat <= 36.0 and -78.9 <= lon <= -78.3:
            is_urban.append(True)
        # Durham area
        elif 35.8 <= lat <= 36.1 and -79.1 <= lon <= -78.7:
            is_urban.append(True)
        # Greensboro area (Guilford)
        elif 35.9 <= lat <= 36.3 and -80.1 <= lon <= -79.6:
            is_urban.append(True)
        else:
            is_urban.append(False)
    
    traffic_data['is_urban'] = is_urban
    traffic_data['AADTT'] = base_aadtt
    
    # Calculate segment properties
    traffic_data['length_miles'] = np.random.uniform(0.5, 5.0, n_segments)  # Segment lengths
    traffic_data['speed_limit'] = 65  # Typical interstate speed
    
    traffic_df = pd.DataFrame(traffic_data)
    traffic_df['travel_time_hours'] = traffic_df['length_miles'] / traffic_df['speed_limit']
    
    # Calculate demand for each truck class using EXACT FHWA formula
    truck_classes = {
        1: {'name': 'Short-Haul Rest Area', 'haul': 'short_haul', 'facility': 'rest_area'},
        2: {'name': 'Short-Haul Truck Stop', 'haul': 'short_haul', 'facility': 'truck_stop'},
        3: {'name': 'Long-Haul Rest Area', 'haul': 'long_haul', 'facility': 'rest_area'},
        4: {'name': 'Long-Haul Truck Stop', 'haul': 'long_haul', 'facility': 'truck_stop'}
    }
    
    print("\nCalculating demand using FHWA MILP formula:")
    
    for k, class_info in truck_classes.items():
        haul_type = class_info['haul']
        facility_type = class_info['facility']
        
        print(f"\nClass {k}: {class_info['name']}")
        
        # Step-by-step FHWA calculation
        # Step 1: Apply seasonal factor
        seasonal_aadtt = traffic_df['AADTT'] * F_S
        
        # Step 2: Apply haul type proportion based on area type
        area_type = traffic_df['is_urban'].map({True: 'urban', False: 'rural'})
        class_proportion = area_type.map(lambda x: P_CLASS[x][haul_type])
        class_traffic = seasonal_aadtt * class_proportion
        
        # Step 3: Apply facility type proportion
        facility_traffic = class_traffic * P_FACILITY[facility_type]
        
        # Step 4: Apply peak hour factor
        peak_traffic = facility_traffic * P_PEAK[haul_type]
        
        # Step 5: Apply travel time and parking factors
        travel_time_factor = traffic_df['travel_time_hours']
        parking_factor = P_PARK[haul_type]
        
        final_demand = peak_traffic * travel_time_factor * parking_factor
        
        traffic_df[f'demand_class_{k}'] = final_demand.fillna(0)
        
        total_class_demand = final_demand.sum()
        print(f"  Total demand: {total_class_demand:.2f} trucks")
        
        # Show the calculation breakdown for understanding
        avg_seasonal = seasonal_aadtt.mean()
        avg_class_prop = class_proportion.mean()
        avg_facility_prop = P_FACILITY[facility_type]
        avg_peak = P_PEAK[haul_type]
        avg_travel = travel_time_factor.mean()
        avg_parking = parking_factor
        
        print(f"  Formula: AADTT({avg_seasonal:.0f}) × Class({avg_class_prop:.3f}) × Facility({avg_facility_prop:.3f}) × Peak({avg_peak:.3f}) × Travel({avg_travel:.3f}h) × Parking({avg_parking:.3f}h)")
    
    # Add higher-demand clusters around major cities using same formula
    charlotte_segments = pd.DataFrame({
        'Latitude': np.random.normal(35.2271, 0.15, 40),
        'Longitude': np.random.normal(-80.8431, 0.15, 40),
        'is_urban': True,
        'AADTT': np.random.exponential(1200, 40),  # Higher urban traffic
        'length_miles': np.random.uniform(1.0, 3.0, 40),
        'speed_limit': 65
    })
    charlotte_segments['travel_time_hours'] = charlotte_segments['length_miles'] / charlotte_segments['speed_limit']
    
    # Apply same FHWA formula to Charlotte segments
    for k, class_info in truck_classes.items():
        haul_type = class_info['haul']
        facility_type = class_info['facility']
        
        seasonal_aadtt = charlotte_segments['AADTT'] * F_S
        class_proportion = P_CLASS['urban'][haul_type]  # All urban
        class_traffic = seasonal_aadtt * class_proportion
        facility_traffic = class_traffic * P_FACILITY[facility_type]
        peak_traffic = facility_traffic * P_PEAK[haul_type]
        final_demand = peak_traffic * charlotte_segments['travel_time_hours'] * P_PARK[haul_type]
        
        charlotte_segments[f'demand_class_{k}'] = final_demand.fillna(0)
    
    # Similar for Raleigh-Durham
    raleigh_segments = pd.DataFrame({
        'Latitude': np.random.normal(35.7796, 0.15, 35),
        'Longitude': np.random.normal(-78.6382, 0.15, 35),
        'is_urban': True,
        'AADTT': np.random.exponential(1000, 35),
        'length_miles': np.random.uniform(1.0, 3.0, 35),
        'speed_limit': 65
    })
    raleigh_segments['travel_time_hours'] = raleigh_segments['length_miles'] / raleigh_segments['speed_limit']
    
    for k, class_info in truck_classes.items():
        haul_type = class_info['haul']
        facility_type = class_info['facility']
        
        seasonal_aadtt = raleigh_segments['AADTT'] * F_S
        class_proportion = P_CLASS['urban'][haul_type]
        class_traffic = seasonal_aadtt * class_proportion
        facility_traffic = class_traffic * P_FACILITY[facility_type]
        peak_traffic = facility_traffic * P_PEAK[haul_type]
        final_demand = peak_traffic * raleigh_segments['travel_time_hours'] * P_PARK[haul_type]
        
        raleigh_segments[f'demand_class_{k}'] = final_demand.fillna(0)
    
    # Combine all segments
    all_segments = pd.concat([traffic_df, charlotte_segments, raleigh_segments], ignore_index=True)
    
    # Calculate and report final demand distribution
    total_demands = {}
    for class_id in range(1, 5):
        total_demands[f'class_{class_id}'] = all_segments[f'demand_class_{class_id}'].sum()
    
    total_all_demand = sum(total_demands.values())
    
    print(f"\n=== FINAL FHWA-BASED DEMAND SUMMARY ===")
    print(f"Total segments: {len(all_segments)}")
    print(f"Urban segments: {all_segments['is_urban'].sum()}")
    print(f"Rural segments: {(~all_segments['is_urban']).sum()}")
    print(f"Total demand: {total_all_demand:.0f} trucks")
    
    for class_id in range(1, 5):
        class_name = truck_classes[class_id]['name']
        demand = total_demands[f'class_{class_id}']
        percentage = demand / total_all_demand * 100
        print(f"  {class_name}: {demand:.0f} trucks ({percentage:.1f}%)")
    
    # Verify the ratios match FHWA expectations
    short_haul_total = total_demands['class_1'] + total_demands['class_2']
    long_haul_total = total_demands['class_3'] + total_demands['class_4']
    rest_area_total = total_demands['class_1'] + total_demands['class_3']
    truck_stop_total = total_demands['class_2'] + total_demands['class_4']
    
    print(f"\nFHWA Ratio Verification:")
    print(f"  Short-haul: {short_haul_total/total_all_demand:.1%}")
    print(f"  Long-haul: {long_haul_total/total_all_demand:.1%}")
    print(f"  Rest area preference: {rest_area_total/total_all_demand:.1%}")
    print(f"  Truck stop preference: {truck_stop_total/total_all_demand:.1%}")
    
    return all_segments

def smart_assign_existing_facility_types(existing_facilities):
    """Enhanced facility type assignment that properly maps real-world facility types"""
    facility_types = {}
    excluded_facilities = []
    
    print("=== ENHANCED EXISTING FACILITY TYPE ASSIGNMENT ===")
    
    # Check available columns
    has_overnight = 'over_night' in existing_facilities.columns
    has_facility_t = 'Facility_T' in existing_facilities.columns
    
    print(f"Available data columns: over_night={has_overnight}, Facility_T={has_facility_t}")
    
    if not has_facility_t:
        print("WARNING: Facility_T column missing. Excluding all existing facilities.")
        return {}, list(existing_facilities.index)
    
    type_counts = {'rest_area': 0, 'truck_stop': 0}
    
    for idx, facility in existing_facilities.iterrows():
        facility_t = facility.get('Facility_T', None)
        
        # Exclude facilities with missing type information
        if pd.isna(facility_t) or facility_t == '':
            excluded_facilities.append(idx)
            continue
        
        # Enhanced classification logic that properly maps facility types
        facility_t_str = str(facility_t).lower()
        
        # Public facilities = rest areas (basic service, government operated)
        if facility_t_str == 'public':
            facility_types[idx] = 'rest_area'
            type_counts['rest_area'] += 1
            
        # Walmart facilities: context-dependent classification
        elif facility_t_str == 'wal-mart':
            if has_overnight:
                over_night = facility.get('over_night', None)
                if pd.isna(over_night) or str(over_night).lower() != 'yes':
                    # No overnight = basic parking = rest area function
                    facility_types[idx] = 'rest_area'
                    type_counts['rest_area'] += 1
                else:
                    # Overnight capability = comprehensive service = truck stop function
                    facility_types[idx] = 'truck_stop'
                    type_counts['truck_stop'] += 1
            else:
                # Default Walmart to rest area (basic parking) if no overnight data
                facility_types[idx] = 'rest_area'
                type_counts['rest_area'] += 1
                
        # All other private facilities = truck stops (comprehensive service, commercially operated)
        # This includes gas stations, truck stops, travel centers, etc.
        else:
            facility_types[idx] = 'truck_stop'
            type_counts['truck_stop'] += 1
    
    print(f"Enhanced facility type assignment results:")
    print(f"  Rest areas (public/basic service): {type_counts['rest_area']}")
    print(f"  Truck stops (private/full service): {type_counts['truck_stop']}")
    print(f"  Excluded (missing data): {len(excluded_facilities)}")
    
    return facility_types, excluded_facilities

def prepare_facilities_for_optimization(candidates_gdf, existing_facilities, traffic_segments):
    """Prepare unified facility dataset for optimization with enhanced proximity analysis"""
    print("Preparing facilities for optimization...")
    
    # Smart assignment of existing facility types
    existing_facility_types, excluded_facilities = smart_assign_existing_facility_types(existing_facilities)
    
    # Filter out excluded existing facilities
    existing_clean = existing_facilities.drop(excluded_facilities).copy()
    print(f"Using {len(existing_clean)} existing facilities after filtering")
    
    # Process existing facilities with proper coordinate handling
    existing_clean['capacity'] = existing_clean.get('Final_Park', 0).fillna(0)
    existing_clean['facility_type'] = 'existing'
    
    # Extract coordinates from geometry with error handling
    try:
        existing_clean['Latitude'] = existing_clean.geometry.y
        existing_clean['Longitude'] = existing_clean.geometry.x
    except Exception as e:
        print(f"Error extracting existing facility coordinates: {e}")
        # Try alternative coordinate extraction
        coords = existing_clean.geometry.apply(lambda geom: (geom.x, geom.y) if hasattr(geom, 'x') else (None, None))
        existing_clean['Longitude'] = coords.apply(lambda x: x[0])
        existing_clean['Latitude'] = coords.apply(lambda x: x[1])
    
    # Validate existing facility coordinates
    existing_lat_range = (existing_clean['Latitude'].min(), existing_clean['Latitude'].max())
    existing_lon_range = (existing_clean['Longitude'].min(), existing_clean['Longitude'].max())
    print(f"Existing facility coordinates: Lat {existing_lat_range}, Lon {existing_lon_range}")
    
    # Create existing facility records
    existing_records = []
    for idx, row in existing_clean.iterrows():
        existing_records.append({
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'capacity': row['capacity'],
            'facility_type': 'existing',
            'original_index': idx,
            'assigned_facility_type': existing_facility_types.get(idx, 'truck_stop')
        })
    
    # Process candidate facilities
    candidate_records = []
    candidate_start_idx = len(existing_records)
    
    print(f"Processing {len(candidates_gdf)} candidate facilities...")
    
    for idx, row in candidates_gdf.iterrows():
        try:
            # Extract coordinates from geometry
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
    
    # Validate candidate coordinates
    if candidate_records:
        candidate_lats = [r['Latitude'] for r in candidate_records]
        candidate_lons = [r['Longitude'] for r in candidate_records]
        candidate_lat_range = (min(candidate_lats), max(candidate_lats))
        candidate_lon_range = (min(candidate_lons), max(candidate_lons))
        print(f"Candidate coordinates: Lat {candidate_lat_range}, Lon {candidate_lon_range}")
    
    # Combine all facilities
    all_facilities_list = existing_records + candidate_records
    all_facilities = pd.DataFrame(all_facilities_list)
    
    print(f"Combined facilities dataset:")
    print(f"  Existing facilities: {len(existing_records)}")
    print(f"  Candidate facilities: {len(candidate_records)}")
    print(f"  Total facilities: {len(all_facilities)}")
    
    # Create facility type mapping for existing facilities
    facility_type_mapping = {}
    for idx, row in all_facilities.iterrows():
        if row['facility_type'] == 'existing':
            facility_type_mapping[idx] = row['assigned_facility_type']
    
    return all_facilities, facility_type_mapping, candidate_start_idx

def analyze_demand_patterns(candidate_lat, candidate_lon, traffic_segments, radius=25.0):
    """Analyze demand patterns around a candidate location to recommend facility type"""
    
    # Calculate distances to all traffic segments
    segment_distances = []
    
    for idx, segment in traffic_segments.iterrows():
        # Simple distance calculation (can be replaced with haversine for accuracy)
        lat_diff = candidate_lat - segment['Latitude']
        lon_diff = candidate_lon - segment['Longitude']
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 69  # Rough miles conversion
        
        if distance <= radius:
            segment_distances.append({
                'distance': distance,
                'demand_class_1': segment['demand_class_1'],  # Rest area short-haul
                'demand_class_2': segment['demand_class_2'],  # Truck stop short-haul  
                'demand_class_3': segment['demand_class_3'],  # Rest area long-haul
                'demand_class_4': segment['demand_class_4']   # Truck stop long-haul
            })
    
    if not segment_distances:
        # No nearby demand segments - default to truck stop for comprehensive coverage
        return {
            'recommended_type': 'truck_stop',
            'confidence': 0.6,
            'reasoning': 'No nearby demand data - truck stop provides comprehensive coverage',
            'demand_analysis': {'rest_area_total': 0, 'truck_stop_total': 0}
        }
    
    # Weight demand by inverse distance (closer segments matter more)
    weighted_rest_area_demand = 0
    weighted_truck_stop_demand = 0
    
    for segment in segment_distances:
        weight = 1 / (1 + segment['distance'])  # Inverse distance weighting
        
        # Rest area demand = classes 1 + 3
        rest_area_demand = (segment['demand_class_1'] + segment['demand_class_3']) * weight
        weighted_rest_area_demand += rest_area_demand
        
        # Truck stop demand = classes 2 + 4  
        truck_stop_demand = (segment['demand_class_2'] + segment['demand_class_4']) * weight
        weighted_truck_stop_demand += truck_stop_demand
    
    # Determine recommendation based on dominant demand type
    total_demand = weighted_rest_area_demand + weighted_truck_stop_demand
    
    if total_demand == 0:
        return {
            'recommended_type': 'truck_stop',
            'confidence': 0.5,
            'reasoning': 'No significant demand detected - truck stop for general coverage',
            'demand_analysis': {'rest_area_total': 0, 'truck_stop_total': 0}
        }
    
    truck_stop_ratio = weighted_truck_stop_demand / total_demand
    rest_area_ratio = weighted_rest_area_demand / total_demand
    
    # Decision logic with confidence scoring
    if truck_stop_ratio > 0.65:  # Increased threshold to be more selective
        return {
            'recommended_type': 'truck_stop',
            'confidence': min(0.9, truck_stop_ratio + 0.1),
            'reasoning': f'Strong truck stop demand ({truck_stop_ratio:.1%} of total)',
            'demand_analysis': {
                'rest_area_total': weighted_rest_area_demand,
                'truck_stop_total': weighted_truck_stop_demand,
                'truck_stop_ratio': truck_stop_ratio
            }
        }
    elif rest_area_ratio > 0.65:  # Increased threshold to be more selective
        return {
            'recommended_type': 'rest_area',  
            'confidence': min(0.9, rest_area_ratio + 0.1),
            'reasoning': f'Strong rest area demand ({rest_area_ratio:.1%} of total)',
            'demand_analysis': {
                'rest_area_total': weighted_rest_area_demand,
                'truck_stop_total': weighted_truck_stop_demand,
                'rest_area_ratio': rest_area_ratio
            }
        }
    elif rest_area_ratio > 0.45:  # Give rest areas more consideration in balanced scenarios
        return {
            'recommended_type': 'rest_area',
            'confidence': 0.6,
            'reasoning': f'Moderate rest area preference ({rest_area_ratio:.1%} vs {truck_stop_ratio:.1%}) - rest area for cost-effective basic service',
            'demand_analysis': {
                'rest_area_total': weighted_rest_area_demand,
                'truck_stop_total': weighted_truck_stop_demand,
                'truck_stop_ratio': truck_stop_ratio,
                'rest_area_ratio': rest_area_ratio
            }
        }
    else:
        # Default to truck stop only when clearly favored
        return {
            'recommended_type': 'truck_stop',
            'confidence': 0.6,
            'reasoning': f'Truck stop preferred ({truck_stop_ratio:.1%} vs {rest_area_ratio:.1%}) - comprehensive service coverage',
            'demand_analysis': {
                'rest_area_total': weighted_rest_area_demand,
                'truck_stop_total': weighted_truck_stop_demand,
                'truck_stop_ratio': truck_stop_ratio,
                'rest_area_ratio': rest_area_ratio
            }
        }

def load_data():
    """Load and prepare all data for optimization with improved error handling"""
    print("Loading and preparing data...")
    
    # Load candidate locations with comprehensive error handling
    candidate_path = os.path.join(script_dir, "../results/composite_prioritization_scores.csv")
    try:
        candidates_df = pd.read_csv(candidate_path)
        print(f"Loaded {len(candidates_df)} candidate locations")
    except Exception as e:
        print(f"Error loading candidates: {e}")
        return None, None, None
   
    # Load existing facilities with fallback options
    existing_path = os.path.join(script_dir, "../datasets/existing_fac_without_weigh_station.shp")
    try:
        existing_facilities = gpd.read_file(existing_path)
        print(f"Loaded {len(existing_facilities)} existing facilities from shapefile")
    except:
        try:
            # Try CSV fallback
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
    
    # Convert candidates to GeoDataFrame with proper coordinate system handling
    if 'geometry' not in candidates_df.columns:
        if 'centroid_x' in candidates_df.columns and 'centroid_y' in candidates_df.columns:
            # Check coordinate system
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
    
    # Ensure consistent coordinate reference system
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
    
    # Load traffic segments with FHWA-derived parameters
    traffic_segments = load_traffic_segments()
    
    # Load county boundary data (optional)
    counties_gdf = load_county_data()
    
    # Add location information to candidates
    candidates_gdf = add_location_info_to_candidates(candidates_gdf, counties_gdf)
    
    # Calculate development costs with capacity capping
    candidates_gdf = calculate_development_costs(candidates_gdf)
    
    return candidates_gdf, existing_facilities, traffic_segments

# DIAGNOSTIC FUNCTIONS

def debug_facility_choice_enhanced(candidate_idx, all_facilities, candidates_gdf, traffic_segments, 
                                 lp_calculator, current_facilities, current_facility_types, 
                                 candidate_start_idx, budget_remaining, proximity_analyzer):
    """Debug version of facility choice optimization with detailed logging"""
    print(f"\n    DEBUG FACILITY CHOICE for candidate {candidate_idx}:")
    
    # Get candidate information
    candidate_original_idx = candidate_idx - candidate_start_idx
    if candidate_original_idx < 0 or candidate_original_idx >= len(candidates_gdf):
        print(f"      ERROR: Invalid candidate index {candidate_original_idx}")
        return None
    
    candidate_row = candidates_gdf.iloc[candidate_original_idx]
    candidate_lat = candidate_row.geometry.y
    candidate_lon = candidate_row.geometry.x
    
    print(f"      Location: ({candidate_lat:.4f}, {candidate_lon:.4f})")
    print(f"      Original capacity: {candidate_row['capacity_value']}")
    print(f"      No service cost: ${candidate_row['cost_no_service']/1e6:.2f}M")
    print(f"      Full service cost: ${candidate_row['cost_full_service']/1e6:.2f}M")
    print(f"      Budget remaining: ${budget_remaining/1e6:.2f}M")
    
    # Check budget constraints
    if candidate_row['cost_no_service'] > budget_remaining and candidate_row['cost_full_service'] > budget_remaining:
        print(f"      REJECTED: Both options exceed budget")
        return None
    
    # Step 1: Analyze local demand patterns
    try:
        print(f"      Analyzing demand patterns...")
        demand_analysis = analyze_demand_patterns(candidate_lat, candidate_lon, traffic_segments)
        print(f"      Demand recommendation: {demand_analysis['recommended_type']} (confidence: {demand_analysis['confidence']:.3f})")
        print(f"      Demand reasoning: {demand_analysis['reasoning']}")
    except Exception as e:
        print(f"      ERROR in demand analysis: {e}")
        return None
    
    # Step 2: Analyze local infrastructure
    try:
        print(f"      Analyzing proximity...")
        proximity_analysis = proximity_analyzer.analyze_local_infrastructure(
            candidate_lat, candidate_lon, current_facility_types
        )
        print(f"      Proximity recommendation: {proximity_analysis['recommended_type']} (confidence: {proximity_analysis['confidence']:.3f})")
        print(f"      Proximity reasoning: {proximity_analysis['reasoning']}")
    except Exception as e:
        print(f"      ERROR in proximity analysis: {e}")
        return None
    
    # Step 3: Integrate analyses
    if demand_analysis['recommended_type'] == proximity_analysis['recommended_type']:
        selected_facility_type = demand_analysis['recommended_type']
        decision_confidence = min(0.95, (demand_analysis['confidence'] + proximity_analysis['confidence']) / 2 + 0.2)
        decision_reasoning = f"Both analyses agree: {demand_analysis['recommended_type']}"
        print(f"      DECISION: Both agree on {selected_facility_type} (confidence: {decision_confidence:.3f})")
    else:
        # Use weighted decision
        demand_weight = 0.6
        proximity_weight = 0.4
        
        if (demand_analysis['confidence'] * demand_weight) > (proximity_analysis['confidence'] * proximity_weight):
            selected_facility_type = demand_analysis['recommended_type']
            decision_confidence = demand_analysis['confidence'] * 0.8
            decision_reasoning = f"Demand analysis preferred: {demand_analysis['recommended_type']}"
        else:
            selected_facility_type = proximity_analysis['recommended_type']
            decision_confidence = proximity_analysis['confidence'] * 0.8
            decision_reasoning = f"Proximity analysis preferred: {proximity_analysis['recommended_type']}"
        
        print(f"      DECISION: Analyses disagree, chose {selected_facility_type} (confidence: {decision_confidence:.3f})")
    
    # Step 4: Calculate costs and capacity
    if selected_facility_type == 'rest_area':
        cost = candidate_row['cost_no_service']
        effective_capacity = candidate_row['capped_capacity_no_service']
        service_level = 'No Service'
    else:
        cost = candidate_row['cost_full_service'] 
        effective_capacity = candidate_row['capped_capacity_full_service']
        service_level = 'Full Service'
    
    print(f"      Selected config: {service_level} {selected_facility_type}")
    print(f"      Cost: ${cost/1e6:.2f}M, Capacity: {effective_capacity}")
    
    # Check budget constraint
    if cost > budget_remaining:
        print(f"      REJECTED: Cost ${cost/1e6:.2f}M > Budget ${budget_remaining/1e6:.2f}M")
        return None
    
    # Step 5: Calculate performance
    try:
        print(f"      Calculating performance...")
        
        # Get current unmet demand
        current_results = lp_calculator.calculate_unmet_demand_lp(current_facilities, current_facility_types)
        current_unmet = current_results.get('unmet_total', 0)
        print(f"      Current unmet demand: {current_unmet:.1f}")
        
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
        
        print(f"      Test unmet demand: {test_unmet:.1f}")
        print(f"      Demand reduction: {demand_reduction:.1f}")
        print(f"      Cost effectiveness: {cost_effectiveness:.6f}")
        
        # Calculate class-specific reductions
        class_reductions = {}
        for class_id in range(1, 5):
            current_class = current_results.get(f'unmet_class_{class_id}', 0)
            test_class = test_results.get(f'unmet_class_{class_id}', 0)
            class_reductions[f'class_{class_id}_reduction'] = max(0, current_class - test_class)
        
        # Calculate composite score
        secondary_factors = [
            candidate_row.get('crash_risk_norm', 0.5),
            candidate_row.get('accessibility_norm', 0.5),
            candidate_row.get('traffic_influx_norm', 0.5),
            candidate_row.get('capacity_norm', 0.5)
        ]
        secondary_score = sum(secondary_factors) / len(secondary_factors)
        
        if current_unmet > 0:
            demand_score = min(1.0, demand_reduction / (current_unmet * 0.1))
        else:
            demand_score = 0
        
        confidence_boost = (decision_confidence - 0.5) * 0.1
        composite_score = 0.5 * demand_score + 0.5 * secondary_score + confidence_boost
        
        print(f"      Demand score: {demand_score:.4f}")
        print(f"      Secondary score: {secondary_score:.4f}")
        print(f"      Confidence boost: {confidence_boost:.4f}")
        print(f"      Composite score: {composite_score:.4f}")
        
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
        print(f"      ERROR calculating performance: {e}")
        import traceback
        traceback.print_exc()
        return None

def diagnostic_unified_budget_optimization(candidates_gdf, existing_facilities, traffic_segments, max_budget):
    """Diagnostic version of unified optimization with extensive debugging"""
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC UNIFIED OPTIMIZATION: ${max_budget/1e6:.0f}M Budget")
    print('='*80)
    
    # DIAGNOSTIC 1: Input validation
    print(f"DIAGNOSTIC 1 - INPUT VALIDATION:")
    print(f"  Candidates: {len(candidates_gdf)}")
    print(f"  Existing facilities: {len(existing_facilities)}")
    print(f"  Traffic segments: {len(traffic_segments)}")
    print(f"  Max budget: ${max_budget:,.0f}")
    
    if len(candidates_gdf) == 0:
        print("  ERROR: No candidate facilities provided!")
        return pd.DataFrame()
    
    # Check for required columns
    required_candidate_cols = ['capacity_value', 'cost_no_service', 'cost_full_service']
    missing_cols = [col for col in required_candidate_cols if col not in candidates_gdf.columns]
    if missing_cols:
        print(f"  ERROR: Missing candidate columns: {missing_cols}")
        return pd.DataFrame()
    
    # DIAGNOSTIC 2: Cost analysis
    print(f"\nDIAGNOSTIC 2 - COST ANALYSIS:")
    print(f"  No service costs: ${candidates_gdf['cost_no_service'].min()/1e6:.2f}M to ${candidates_gdf['cost_no_service'].max()/1e6:.2f}M")
    print(f"  Full service costs: ${candidates_gdf['cost_full_service'].min()/1e6:.2f}M to ${candidates_gdf['cost_full_service'].max()/1e6:.2f}M")
    print(f"  Capacity range: {candidates_gdf['capacity_value'].min():.0f} to {candidates_gdf['capacity_value'].max():.0f}")
    
    # Check if any facilities are within budget
    affordable_no_service = (candidates_gdf['cost_no_service'] <= max_budget).sum()
    affordable_full_service = (candidates_gdf['cost_full_service'] <= max_budget).sum()
    print(f"  Affordable no service facilities: {affordable_no_service}")
    print(f"  Affordable full service facilities: {affordable_full_service}")
    
    if affordable_no_service == 0 and affordable_full_service == 0:
        print("  ERROR: No facilities are affordable within budget!")
        return pd.DataFrame()
    
    # Prepare facilities for optimization
    try:
        print(f"\nDIAGNOSTIC 3 - FACILITY PREPARATION:")
        all_facilities, existing_facility_types, candidate_start_idx = prepare_facilities_for_optimization(
            candidates_gdf, existing_facilities, traffic_segments
        )
        print(f"  ✓ Total facilities prepared: {len(all_facilities)}")
        print(f"  ✓ Existing facilities: {candidate_start_idx}")
        print(f"  ✓ Candidate facilities: {len(all_facilities) - candidate_start_idx}")
        print(f"  ✓ Existing facility types: {len(existing_facility_types)}")
        
        # Show existing facility type distribution
        if existing_facility_types:
            type_counts = {}
            for ftype in existing_facility_types.values():
                type_counts[ftype] = type_counts.get(ftype, 0) + 1
            print(f"  ✓ Existing type distribution: {type_counts}")
        
    except Exception as e:
        print(f"  ERROR in facility preparation: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # Initialize calculators
    try:
        print(f"\nDIAGNOSTIC 4 - CALCULATOR INITIALIZATION:")
        lp_calculator = UnmetDemandCalculator(traffic_segments, all_facilities)
        proximity_analyzer = ProximityAnalyzer(all_facilities)
        print(f"  ✓ LP calculator initialized")
        print(f"  ✓ Proximity analyzer initialized")
        
        # Test initial unmet demand calculation
        print(f"  Testing baseline demand calculation...")
        baseline_results = lp_calculator.calculate_unmet_demand_lp([], {})
        baseline_unmet = baseline_results.get('unmet_total', 0)
        print(f"  ✓ Baseline unmet demand: {baseline_unmet:.1f}")
        
        if baseline_unmet <= 0:
            print(f"  WARNING: Baseline unmet demand is {baseline_unmet}, this seems wrong!")
        
    except Exception as e:
        print(f"  ERROR in calculator initialization: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # Initialize optimization state
    print(f"\nDIAGNOSTIC 5 - OPTIMIZATION INITIALIZATION:")
    current_facilities = [idx for idx in range(candidate_start_idx)]
    current_facility_types = existing_facility_types.copy()
    remaining_candidates = list(range(candidate_start_idx, len(all_facilities)))
    
    print(f"  ✓ Starting facilities: {len(current_facilities)}")
    print(f"  ✓ Remaining candidates: {len(remaining_candidates)}")
    
    # Calculate initial state
    try:
        initial_results = lp_calculator.calculate_unmet_demand_lp(current_facilities, current_facility_types)
        initial_unmet = initial_results.get('unmet_total', 0)
        print(f"  ✓ Initial unmet demand with existing facilities: {initial_unmet:.1f}")
        print(f"  ✓ Demand reduction from existing: {baseline_unmet - initial_unmet:.1f}")
    except Exception as e:
        print(f"  ERROR calculating initial state: {e}")
        initial_unmet = baseline_unmet
    
    # Track optimization progress
    selections = []
    cumulative_budget = 0
    cumulative_capacity = 0
    selection_count = 0
    
    print(f"\nDIAGNOSTIC 6 - OPTIMIZATION LOOP:")
    print(f"Starting main optimization loop...")
    
    # Diagnostic counters
    iterations_without_selection = 0
    max_iterations = min(100, len(remaining_candidates))  # Safety limit
    
    # Main optimization loop with enhanced debugging
    for iteration in range(max_iterations):
        if not remaining_candidates or cumulative_budget >= max_budget:
            print(f"  Loop exit: remaining_candidates={len(remaining_candidates)}, budget_used=${cumulative_budget/1e6:.1f}M")
            break
        
        budget_remaining = max_budget - cumulative_budget
        
        print(f"\n  --- ITERATION {iteration + 1} ---")
        print(f"  Budget remaining: ${budget_remaining/1e6:.2f}M")
        print(f"  Candidates remaining: {len(remaining_candidates)}")
        print(f"  Facilities selected so far: {selection_count}")
        
        # Test first few candidates in detail
        candidate_evaluations = []
        candidates_tested = 0
        candidates_affordable = 0
        candidates_valid_config = 0
        
        for i, candidate_idx in enumerate(remaining_candidates[:10]):  # Test first 10 in detail
            candidates_tested += 1
            
            # Get candidate info
            candidate_original_idx = candidate_idx - candidate_start_idx
            candidate_row = candidates_gdf.iloc[candidate_original_idx]
            
            # Check affordability
            min_cost = min(candidate_row['cost_no_service'], candidate_row['cost_full_service'])
            if min_cost <= budget_remaining:
                candidates_affordable += 1
                
                try:
                    config = debug_facility_choice_enhanced(
                        candidate_idx, all_facilities, candidates_gdf, traffic_segments,
                        lp_calculator, current_facilities, current_facility_types,
                        candidate_start_idx, budget_remaining, proximity_analyzer
                    )
                    
                    if config is not None:
                        candidates_valid_config += 1
                        candidate_evaluations.append({
                            'candidate_idx': candidate_idx,
                            'config': config
                        })
                        
                        # Show details for first few candidates
                        if i < 3:
                            print(f"    Candidate {i+1}: Type={config['facility_type']}, Cost=${config['cost']/1e6:.2f}M, Score={config['composite_score']:.4f}")
                    else:
                        if i < 3:
                            print(f"    Candidate {i+1}: Returned None (likely budget/evaluation issue)")
                            
                except Exception as e:
                    print(f"    ERROR evaluating candidate {i+1}: {e}")
            else:
                if i < 3:
                    print(f"    Candidate {i+1}: Too expensive (${min_cost/1e6:.2f}M > ${budget_remaining/1e6:.2f}M)")
        
        print(f"  Diagnostic summary: {candidates_tested} tested, {candidates_affordable} affordable, {candidates_valid_config} valid configs")
        
        # If we tested only first 10, evaluate all remaining quickly
        if len(remaining_candidates) > 10:
            print(f"  Evaluating remaining {len(remaining_candidates) - 10} candidates...")
            for candidate_idx in remaining_candidates[10:]:
                try:
                    config = debug_facility_choice_enhanced(
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
                    continue  # Skip failed evaluations
        
        print(f"  Total valid configurations: {len(candidate_evaluations)}")
        
        # Select the best candidate
        if not candidate_evaluations:
            print(f"  No valid candidates found in iteration {iteration + 1}")
            iterations_without_selection += 1
            if iterations_without_selection >= 3:
                print(f"  Stopping: 3 consecutive iterations without valid candidates")
                break
            continue
        
        # Reset no-selection counter
        iterations_without_selection = 0
        
        # Sort by composite score and select best
        candidate_evaluations.sort(key=lambda x: x['config']['composite_score'], reverse=True)
        best_candidate = candidate_evaluations[0]
        
        selected_idx = best_candidate['candidate_idx']
        best_config = best_candidate['config']
        
        print(f"  SELECTED: Index {selected_idx}")
        print(f"    Type: {best_config['facility_type']}")
        print(f"    Service: {best_config['service_level']}")
        print(f"    Cost: ${best_config['cost']/1e6:.2f}M")
        print(f"    Score: {best_config['composite_score']:.4f}")
        print(f"    Demand reduction: {best_config['demand_reduction']:.1f}")
        print(f"    Confidence: {best_config['decision_confidence']:.3f}")
        
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
        
        # Record selection
        selection_record = {
            'Selection_Order': selection_count,
            'FID': candidate_row.get('FID', selected_idx),
            'Facility_Name': candidate_row.get('ComplexNam', f'Facility_{selected_idx}'),
            'County': candidate_row.get('county', 'Unknown'),
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
        
        selections.append(selection_record)
        
        print(f"    Cumulative: ${cumulative_budget/1e6:.2f}M budget, {cumulative_capacity} capacity")
        
        # Check for low scores (stopping criterion)
        if best_config['composite_score'] < 0.001:
            print(f"  Stopping: Score {best_config['composite_score']:.6f} below threshold 0.001")
            break
    
    # Create results DataFrame
    results_df = pd.DataFrame(selections)
    
    print(f"\nDIAGNOSTIC 7 - FINAL RESULTS:")
    print(f"  Facilities selected: {len(results_df)}")
    if len(results_df) > 0:
        print(f"  Budget used: ${cumulative_budget/1e6:.2f}M ({cumulative_budget/max_budget*100:.1f}%)")
        print(f"  Capacity added: {cumulative_capacity:,}")
        print(f"  Average score: {results_df['Composite_Score'].mean():.4f}")
        
        # Service and type distribution
        if 'Service_Level' in results_df.columns:
            service_dist = results_df['Service_Level'].value_counts().to_dict()
            print(f"  Service distribution: {service_dist}")
        
        if 'Facility_Type' in results_df.columns:
            type_dist = results_df['Facility_Type'].value_counts().to_dict()
            print(f"  Type distribution: {type_dist}")
        
        # Show first few selections
        print(f"\nFirst 5 selections:")
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            print(f"  {i+1}. {row['Facility_Name']} - {row['Facility_Type']} - ${row['Cost']/1e6:.2f}M - Score: {row['Composite_Score']:.4f}")
    else:
        print(f"  ERROR: No facilities were selected!")
        
        # Diagnose why no facilities were selected
        print(f"\nDIAGNOSTIC - Why no facilities selected:")
        print(f"  Total candidates: {len(candidates_gdf)}")
        print(f"  Budget: ${max_budget/1e6:.0f}M")
        print(f"  Cheapest no-service: ${candidates_gdf['cost_no_service'].min()/1e6:.2f}M")
        print(f"  Cheapest full-service: ${candidates_gdf['cost_full_service'].min()/1e6:.2f}M")
        
        # Check first candidate manually
        if len(candidates_gdf) > 0:
            first_candidate = candidates_gdf.iloc[0]
            print(f"  First candidate cost analysis:")
            print(f"    No service: ${first_candidate['cost_no_service']/1e6:.2f}M")
            print(f"    Full service: ${first_candidate['cost_full_service']/1e6:.2f}M")
            print(f"    Capacity: {first_candidate['capacity_value']}")
    
    return results_df

def create_diagnostic_plots(results, budget_scenario, output_folder):
    """Create diagnostic plots for the optimization results"""
    
    if len(results) == 0:
        print("No results to plot")
        return
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Diagnostic Analysis - ${budget_scenario/1e6:.0f}M Budget Scenario', fontsize=16, fontweight='bold')
        
        # Plot 1: Facilities vs Budget (Step Plot)
        ax1.step(results['Cumulative_Budget'] / 1e6, results['Selection_Order'], 
                where='post', linewidth=2.5, color='blue', marker='o', markersize=4)
        ax1.set_xlabel('Budget ($ Millions)')
        ax1.set_ylabel('Number of Facilities')
        ax1.set_title('Facilities Selected vs Budget', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Capacity vs Budget (Step Plot)
        ax2.step(results['Cumulative_Budget'] / 1e6, results['Cumulative_Capacity'], 
                where='post', linewidth=2.5, color='green', marker='s', markersize=4)
        ax2.set_xlabel('Budget ($ Millions)')
        ax2.set_ylabel('Cumulative Capacity (Spaces)')
        ax2.set_title('Capacity Added vs Budget', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Facility Type Distribution
        if 'Facility_Type' in results.columns:
            type_counts = results['Facility_Type'].value_counts()
            colors = ['lightcoral', 'lightblue']
            wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax3.set_title('Facility Type Distribution', fontweight='bold')
        
        # Plot 4: Composite Score Evolution
        ax4.plot(results['Selection_Order'], results['Composite_Score'], 
                'o-', linewidth=2, markersize=6, color='red')
        ax4.set_xlabel('Selection Order')
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Composite Score Evolution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_folder, f"diagnostic_analysis_{budget_scenario/1e6:.0f}M.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Diagnostic plot saved to: {plot_file}")
        
        # Create demand class analysis if data exists
        create_demand_class_diagnostic_plots(results, budget_scenario, output_folder)
        
    except Exception as e:
        print(f"Error creating diagnostic plots: {e}")


def create_demand_class_diagnostic_plots(results, budget_scenario, output_folder):
    """Create demand class specific step plots"""
    
    try:
        # Check if demand class columns exist
        class_columns = [col for col in results.columns if col.startswith('class_') and col.endswith('_reduction')]
        
        if not class_columns:
            print("No demand class reduction data found for plotting")
            return
        
        # Define demand class information
        class_info = {
            'class_1_reduction': {'name': 'Short-Haul Rest Area', 'color': '#1f77b4', 'linestyle': '-'},
            'class_2_reduction': {'name': 'Short-Haul Truck Stop', 'color': '#ff7f0e', 'linestyle': '--'},
            'class_3_reduction': {'name': 'Long-Haul Rest Area', 'color': '#2ca02c', 'linestyle': '-.'},
            'class_4_reduction': {'name': 'Long-Haul Truck Stop', 'color': '#d62728', 'linestyle': ':'}
        }
        
        # Create figure for demand class analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Demand Class Analysis - ${budget_scenario/1e6:.0f}M Budget', fontsize=16, fontweight='bold')
        
        # Plot 1: Facilities serving each class vs budget
        ax1.set_title('Facilities Serving Each Demand Class vs Budget', fontsize=12, fontweight='bold')
        
        for class_col in class_columns:
            if class_col in class_info and class_col in results.columns:
                # Count facilities that serve this class (have > 0 reduction)
                facilities_serving_class = (results[class_col] > 0).cumsum()
                
                ax1.step(results['Cumulative_Budget'] / 1e6, facilities_serving_class,
                        where='post', linewidth=2.5, 
                        color=class_info[class_col]['color'],
                        linestyle=class_info[class_col]['linestyle'],
                        label=class_info[class_col]['name'])
        
        ax1.set_xlabel('Budget ($ Millions)')
        ax1.set_ylabel('Number of Facilities Serving Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative demand reduction by class vs budget
        ax2.set_title('Cumulative Demand Reduction by Class vs Budget', fontsize=12, fontweight='bold')
        
        for class_col in class_columns:
            if class_col in class_info and class_col in results.columns:
                cumulative_reduction = results[class_col].cumsum()
                
                ax2.step(results['Cumulative_Budget'] / 1e6, cumulative_reduction,
                        where='post', linewidth=2.5,
                        color=class_info[class_col]['color'],
                        linestyle=class_info[class_col]['linestyle'],
                        label=class_info[class_col]['name'])
        
        ax2.set_xlabel('Budget ($ Millions)')
        ax2.set_ylabel('Cumulative Demand Reduction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total demand reduction by class (bar chart)
        ax3.set_title('Total Demand Reduction by Class', fontsize=12, fontweight='bold')
        
        class_names = []
        total_reductions = []
        colors = []
        
        for class_col in class_columns:
            if class_col in class_info and class_col in results.columns:
                class_names.append(class_info[class_col]['name'])
                total_reductions.append(results[class_col].sum())
                colors.append(class_info[class_col]['color'])
        
        if class_names:
            bars = ax3.bar(range(len(class_names)), total_reductions, color=colors, alpha=0.8)
            ax3.set_xticks(range(len(class_names)))
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
            ax3.set_ylabel('Total Demand Reduction')
            
            # Add value labels on bars
            for bar, value in zip(bars, total_reductions):
                if value > 0:
                    ax3.annotate(f'{value:.0f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom')
        
        # Plot 4: Demand reduction efficiency per facility
        ax4.set_title('Demand Reduction per Facility by Class', fontsize=12, fontweight='bold')
        
        efficiency_values = []
        for class_col in class_columns:
            if class_col in class_info and class_col in results.columns:
                total_reduction = results[class_col].sum()
                facilities_serving = (results[class_col] > 0).sum()
                efficiency = total_reduction / facilities_serving if facilities_serving > 0 else 0
                efficiency_values.append(efficiency)
        
        if class_names and efficiency_values:
            bars = ax4.bar(range(len(class_names)), efficiency_values, color=colors, alpha=0.8)
            ax4.set_xticks(range(len(class_names)))
            ax4.set_xticklabels(class_names, rotation=45, ha='right')
            ax4.set_ylabel('Avg Demand Reduction per Facility')
            
            # Add value labels
            for bar, value in zip(bars, efficiency_values):
                if value > 0:
                    ax4.annotate(f'{value:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the demand class plot
        plot_file = os.path.join(output_folder, f"demand_class_analysis_{budget_scenario/1e6:.0f}M.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Demand class analysis plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"Error creating demand class plots: {e}")


def create_summary_report(results, budget_scenario, output_folder):
    """Create a text summary report of the optimization results"""
    
    try:
        report_file = os.path.join(output_folder, f"optimization_summary_{budget_scenario/1e6:.0f}M.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"OPTIMIZATION SUMMARY REPORT - ${budget_scenario/1e6:.0f}M BUDGET\n")
            f.write("="*80 + "\n\n")
            
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
                f.write("Check diagnostic output for issues\n")
        
        print(f"Summary report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error creating summary report: {e}")

def main_diagnostic():
    """Diagnostic version of main function"""
    print("=" * 80)
    print("DIAGNOSTIC UNIFIED FACILITY OPTIMIZATION SYSTEM")
    print("=" * 80)
    
    try:
        # Load data
        print(f"\n{'='*60}")
        print("DATA LOADING AND PREPARATION")
        print('='*60)
        
        candidates_gdf, existing_facilities, traffic_segments = load_data()
        
        if candidates_gdf is None or existing_facilities is None or traffic_segments is None:
            print("ERROR: Failed to load required data. Exiting.")
            return
        
        # Show current working directory and output paths
        print(f"\nFile paths and locations:")
        print(f"  Script location: {os.path.abspath(__file__)}")
        print(f"  Current working directory: {os.getcwd()}")
        
        # Create output folders
        output_folder = os.path.join(script_dir, "../results/diagnostic_optimization")
        figures_folder = os.path.join(output_folder, "figures")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(figures_folder, exist_ok=True)
        
        print(f"  Results will be saved to: {os.path.abspath(output_folder)}")
        print(f"  Figures will be saved to: {os.path.abspath(figures_folder)}")
        
        # Run optimization for BOTH budget scenarios
        results_dict = {}
        
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            print(f"\n{'='*80}")
            print(f"RUNNING DIAGNOSTIC OPTIMIZATION: {scenario_name.upper()} SCENARIO (${budget/1e6:.0f}M BUDGET)")
            print('='*80)
            
            scenario_results = diagnostic_unified_budget_optimization(
                candidates_gdf, existing_facilities, traffic_segments, budget
            )
            
            # Store results
            results_dict[scenario_name] = scenario_results
            
            # Save CSV results for this scenario
            output_file = os.path.join(output_folder, f"diagnostic_optimization_{scenario_name}_{budget/1e6:.0f}M.csv")
            scenario_results.to_csv(output_file, index=False)
            print(f"\nCSV results saved to: {output_file}")
            
            # Create plots for this scenario
            print(f"\nCreating visualizations for {scenario_name} scenario...")
            create_diagnostic_plots(scenario_results, budget, figures_folder)
            create_summary_report(scenario_results, budget, output_folder)
            
            # Scenario summary
            print(f"\n{'='*60}")
            print(f"{scenario_name.upper()} SCENARIO SUMMARY (${budget/1e6:.0f}M)")
            print('='*60)
            
            if len(scenario_results) > 0:
                print(f"✓ SUCCESS: Selected {len(scenario_results)} facilities")
                print(f"✓ Budget used: ${scenario_results['Cumulative_Budget'].iloc[-1]/1e6:.2f}M ({scenario_results['Cumulative_Budget'].iloc[-1]/budget*100:.1f}%)")
                print(f"✓ Capacity added: {scenario_results['Cumulative_Capacity'].iloc[-1]:,}")
                print(f"✓ Average score: {scenario_results['Composite_Score'].mean():.4f}")
                
                # Show service/type distribution
                if 'Service_Level' in scenario_results.columns:
                    service_dist = scenario_results['Service_Level'].value_counts()
                    print(f"✓ Service levels: {dict(service_dist)}")
                
                if 'Facility_Type' in scenario_results.columns:
                    type_dist = scenario_results['Facility_Type'].value_counts()
                    print(f"✓ Facility types: {dict(type_dist)}")
                    
                # Show top 5 facilities for this scenario
                print(f"\nTop 5 Selected Facilities:")
                for i in range(min(5, len(scenario_results))):
                    row = scenario_results.iloc[i]
                    print(f"  {i+1}. {row['Facility_Name'][:25]:25s} | {row['County'][:12]:12s} | {row['Facility_Type']:10s} | ${row['Cost']/1e6:5.2f}M")
                
            else:
                print("✗ FAILED: No facilities selected")
                print("Check the diagnostic output above for specific issues")
        
        # Create comparative analysis between scenarios
        print(f"\n{'='*80}")
        print("CREATING COMPARATIVE ANALYSIS")
        print('='*80)
        
        create_comparative_analysis(results_dict, figures_folder)
        
        # Final summary comparing both scenarios
        print(f"\n{'='*80}")
        print("FINAL COMPARATIVE SUMMARY")
        print('='*80)
        
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            results = results_dict[scenario_name]
            print(f"\n{scenario_name.upper()} SCENARIO (${budget/1e6:.0f}M):")
            
            if len(results) > 0:
                print(f"  ✓ Facilities: {len(results)}")
                print(f"  ✓ Budget used: ${results['Cumulative_Budget'].iloc[-1]/1e6:.2f}M ({results['Cumulative_Budget'].iloc[-1]/budget*100:.1f}%)")
                print(f"  ✓ Capacity: {results['Cumulative_Capacity'].iloc[-1]:,}")
                print(f"  ✓ Avg score: {results['Composite_Score'].mean():.4f}")
                
                # Facility type breakdown
                if 'Facility_Type' in results.columns:
                    type_counts = results['Facility_Type'].value_counts()
                    for ftype, count in type_counts.items():
                        print(f"    • {ftype.replace('_', ' ').title()}: {count}")
            else:
                print(f"  ✗ No facilities selected")
        
        # List all saved files
        print(f"\n{'='*80}")
        print("ALL SAVED FILES")
        print('='*80)
        
        print(f"📊 CSV Results:")
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            csv_file = f"diagnostic_optimization_{scenario_name}_{budget/1e6:.0f}M.csv"
            print(f"  • {csv_file}")
        
        print(f"\n📈 Diagnostic Plots:")
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            main_plot = f"diagnostic_analysis_{budget/1e6:.0f}M.png"
            class_plot = f"demand_class_analysis_{budget/1e6:.0f}M.png"
            print(f"  • {main_plot}")
            print(f"  • {class_plot}")
        print(f"  • comparative_analysis.png")
        
        print(f"\n📄 Summary Reports:")
        for scenario_name, budget in BUDGET_SCENARIOS.items():
            report_file = f"optimization_summary_{budget/1e6:.0f}M.txt"
            print(f"  • {report_file}")
        
        print(f"\n📁 All files located in: {os.path.abspath(output_folder)}")
        print(f"📁 All plots located in: {os.path.abspath(figures_folder)}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in diagnostic execution: {e}")
        import traceback
        traceback.print_exc()


def create_comparative_analysis(results_dict, figures_folder):
    """Create comparative analysis plots between budget scenarios"""
    
    try:
        current_results = results_dict.get('current', pd.DataFrame())
        expanded_results = results_dict.get('expanded', pd.DataFrame())
        
        # Create comparative figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparative Analysis: $175M vs $1B Budget Scenarios', fontsize=16, fontweight='bold')
        
        # Plot 1: Facilities vs Budget comparison
        ax1.set_title('Facilities Selected vs Budget', fontsize=12, fontweight='bold')
        
        if len(current_results) > 0:
            ax1.step(current_results['Cumulative_Budget'] / 1e6, current_results['Selection_Order'], 
                    where='post', linewidth=2.5, color='blue', label='$175M Budget', marker='o', markersize=4)
        
        if len(expanded_results) > 0:
            ax1.step(expanded_results['Cumulative_Budget'] / 1e6, expanded_results['Selection_Order'], 
                    where='post', linewidth=2.5, color='red', label='$1B Budget', marker='s', markersize=4)
        
        ax1.set_xlabel('Budget ($ Millions)')
        ax1.set_ylabel('Number of Facilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Capacity vs Budget comparison
        ax2.set_title('Cumulative Capacity vs Budget', fontsize=12, fontweight='bold')
        
        if len(current_results) > 0:
            ax2.step(current_results['Cumulative_Budget'] / 1e6, current_results['Cumulative_Capacity'], 
                    where='post', linewidth=2.5, color='blue', label='$175M Budget', marker='o', markersize=4)
        
        if len(expanded_results) > 0:
            ax2.step(expanded_results['Cumulative_Budget'] / 1e6, expanded_results['Cumulative_Capacity'], 
                    where='post', linewidth=2.5, color='red', label='$1B Budget', marker='s', markersize=4)
        
        ax2.set_xlabel('Budget ($ Millions)')
        ax2.set_ylabel('Cumulative Capacity (Spaces)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Facility type comparison
        ax3.set_title('Facility Type Distribution Comparison', fontsize=12, fontweight='bold')
        
        # Prepare data for comparison
        scenarios = ['$175M Budget', '$1B Budget']
        rest_area_counts = []
        truck_stop_counts = []
        
        for results in [current_results, expanded_results]:
            if len(results) > 0 and 'Facility_Type' in results.columns:
                type_counts = results['Facility_Type'].value_counts()
                rest_area_counts.append(type_counts.get('rest_area', 0))
                truck_stop_counts.append(type_counts.get('truck_stop', 0))
            else:
                rest_area_counts.append(0)
                truck_stop_counts.append(0)
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, rest_area_counts, width, label='Rest Areas', alpha=0.8, color='lightblue')
        bars2 = ax3.bar(x + width/2, truck_stop_counts, width, label='Truck Stops', alpha=0.8, color='lightcoral')
        
        ax3.set_xlabel('Budget Scenario')
        ax3.set_ylabel('Number of Facilities')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom')
        
        # Plot 4: Cost effectiveness comparison
        ax4.set_title('Cost Effectiveness Evolution', fontsize=12, fontweight='bold')
        
        if len(current_results) > 0:
            cost_eff_current = current_results['Cumulative_Capacity'] / (current_results['Cumulative_Budget'] / 1e6)
            ax4.plot(current_results['Selection_Order'], cost_eff_current, 
                    'b-o', linewidth=2.5, markersize=4, label='$175M Budget')
        
        if len(expanded_results) > 0:
            cost_eff_expanded = expanded_results['Cumulative_Capacity'] / (expanded_results['Cumulative_Budget'] / 1e6)
            ax4.plot(expanded_results['Selection_Order'], cost_eff_expanded, 
                    'r-s', linewidth=2.5, markersize=4, label='$1B Budget')
        
        ax4.set_xlabel('Selection Order')
        ax4.set_ylabel('Capacity per Million Dollars')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparative plot
        plot_file = os.path.join(figures_folder, "comparative_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparative analysis plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"Error creating comparative analysis: {e}")

if __name__ == "__main__":
    main_diagnostic()