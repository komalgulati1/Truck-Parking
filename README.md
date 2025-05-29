# Truck Facility Location Optimization System

A Python-based iterative optimization system for truck facility location planning using multi-criteria decision analysis and continuous budget allocation.

## Overview

This system optimizes the placement of truck facilities by iteratively selecting the best locations based on composite scoring until a maximum budget is reached. It supports both no-service and full-service facilities with different proximity strategies and cost structures.

## Features

- Iterative facility selection based on composite scoring
- Two facility types: no-service ($0.8M-$1.1M) and full-service ($11M-$13M)
- Continuous budget allocation up to $175M
- Proximity-based selection strategies
- Real-time score recalculation after each selection
- Comprehensive output generation and visualization

## Requirements

```python
pandas
geopandas
numpy
matplotlib
shapely
```

## Project Structure

```
project/
├── scripts/
│   └── iterate.py                      # Main optimization script
├── datasets/
│   ├── existing_fac_without_weigh_station.shp
│   └── finalized_shortlisted_candidates.shp
├── results/
│   ├── composite_prioritization_scores.csv
│   ├── selection_order/
│   ├── continuous_selection/
│   ├── iterations/
│   ├── figures/
│   └── appendices/
└── README.md
```

## Installation

1. Install required dependencies:
```bash
pip install pandas geopandas numpy matplotlib shapely
```

2. Ensure your data files are in the correct locations as shown in the project structure.

## Usage

Run the optimization:
```bash
python iterate.py
```

## Input Data Requirements

### Composite Prioritization Scores CSV
Required columns:
- `FID`: Unique facility identifier
- `ComplexNam`: Facility name
- `centroid_x`, `centroid_y`: Facility coordinates
- `capacity_value`: Facility capacity in spaces
- `traffic_influx_norm`: Normalized traffic score (0-1)
- `crash_risk_norm`: Normalized crash risk score (0-1)
- `accessibility_norm`: Normalized accessibility score (0-1)
- `capacity_norm`: Normalized capacity score (0-1)

### Shapefiles
- `existing_fac_without_weigh_station.shp`: Existing facility locations
- `finalized_shortlisted_candidates.shp`: Candidate facility metadata

## Configuration Parameters

### Cost Structure
```python
# No-Service Facilities
SITE_PREP_NO_SERVICE = 200000      # $200K base preparation
COST_PER_SPACE_NO_SERVICE = 10000  # $10K per parking space

# Full-Service Facilities
SITE_PREP_FULL_SERVICE = 7000000   # $7M base preparation
COST_PER_SPACE_FULL_SERVICE = 67000 # $67K per parking space
```

### Scoring Weights
```python
WEIGHTS = {
    'traffic_influx': 0.25,     # Traffic volume weight
    'crash_risk': 0.25,        # Safety considerations
    'accessibility': 0.25,     # Location accessibility
    'capacity': 0.25           # Facility capacity
}
```

### Budget Limits
```python
MAX_BUDGET = 175e6  # $175 million maximum budget
```

## Algorithm Logic

### Iterative Selection Process
1. Calculate proximity scores for all candidate locations
2. Update composite scores incorporating proximity (20% weight)
3. Select highest-scoring candidate within budget constraints
4. Add selected facility to existing facility network
5. Recalculate all proximity scores and composite rankings
6. Repeat until budget exhausted or no viable candidates remain

### Proximity Strategies
- **No-Service Facilities**: Prefer locations closer to existing facilities for network efficiency
- **Full-Service Facilities**: Prefer locations farther from existing facilities for regional coverage

### Facility Filtering
Automatically excludes inappropriate facility types:
- DMV offices
- License centers
- Welcome centers
- Municipal buildings
- Courthouses
- Rest areas

## Output Files

### Selection Results
- `no_service_selection_order_readable.csv`: Complete no-service facility selection sequence
- `full_service_selection_order_readable.csv`: Complete full-service facility selection sequence

### Summary Tables
- `budget_summary_statistics.csv`: Facility counts and capacity at different budget levels
- `no_service_continuous_selection.csv`: Detailed no-service selections
- `full_service_continuous_selection.csv`: Detailed full-service selections

### Visualizations
- `continuous_step_plot.png/pdf`: Budget allocation visualization showing facility count vs. budget

### Iteration Tracking
- Individual CSV files for each selection iteration saved in `results/iterations/`

### Appendices
- `Appendix_NO_SERVICE_Selected_Facilities.csv`: Facilities selected at each budget level
- `Appendix_FULL_SERVICE_Selected_Facilities.csv`: Budget-based facility selection matrix

## Expected Results

### No-Service Strategy
- Cost range: $0.8M - $1.1M per facility
- Typical output: 160+ facilities at $175M budget
- Strategy: Maximize network coverage near existing infrastructure

### Full-Service Strategy
- Cost range: $11M - $13M per facility
- Typical output: 13-15 facilities at $175M budget
- Strategy: Regional hub placement for comprehensive service coverage

## Customization

### Adjusting Proximity Weight
```python
# In update_composite_score() function
proximity_weight = 0.2  # Modify between 0.1-0.4 for different emphasis
```

### Modifying Excluded Facility Types
```python
EXCLUDED_FACILITY_TYPES = [
    "DMV", "License", "Welcome Center", 
    "Municipal", "Courthouse", "Rest Area"
    # Add or remove facility types as needed
]
```

## Troubleshooting

### Common Issues

**Missing capacity_value column**
```
Error: 'capacity_value' field not found
```
Solution: Ensure the composite scores CSV contains the required `capacity_value` column.

**Shapefile loading errors**
```
Error loading finalized_shortlisted_candidates.shp
```
Solution: Verify all shapefile components (.shp, .shx, .dbf, .prj) are present and accessible.

**Empty facility selections**
Solution: Check that facility names are not empty, "0", or filtered out by facility type exclusions.

### Performance Optimization
- For datasets with 500+ candidates, monitor memory usage during proximity calculations
- Consider processing subsets for very large candidate pools
- Use `.copy()` when modifying DataFrames to avoid pandas warnings

## Technical Notes

- The system uses EPSG:4326 coordinate reference system
- Proximity calculations use Euclidean distance
- All monetary values are in USD
- Output coordinates are preserved for GIS integration

