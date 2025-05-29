# Facility Location Optimization System

A Python-based iterative optimization system for selecting optimal locations for transportation facilities (no-service and full-service) based on composite scoring, budget constraints, and proximity analysis.

## Overview

This system performs continuous budget allocation analysis to determine the optimal sequence of facility construction within a maximum budget of $175 million. It considers multiple factors including traffic patterns, crash risk, accessibility, capacity, and proximity to existing facilities.

## Features

- **Dual Facility Types**: Supports both no-service and full-service facility optimization
- **Iterative Selection**: Continuously selects facilities based on composite scores until budget is exhausted
- **Proximity Analysis**: Calculates strategic placement relative to existing facilities
- **Cost Modeling**: Dynamic cost calculation based on facility capacity
- **Comprehensive Reporting**: Generates detailed selection tables, plots, and appendices

## Requirements

### Dependencies
```python
pandas
geopandas
numpy
matplotlib
shapely
```

### Installation
```bash
pip install pandas geopandas numpy matplotlib shapely
```

## Data Requirements

### Input Files
The system expects the following files in the `datasets/` directory:

1. **`composite_prioritization_scores.csv`**
   - Candidate locations with composite scores
   - Required columns: `ComplexNam`, `centroid_x`, `centroid_y`, `capacity_value`, `FID`
   - Normalized scoring columns: `traffic_influx_norm`, `crash_risk_norm`, `accessibility_norm`, `capacity_norm`

2. **`existing_fac_without_weigh_station.shp`**
   - Existing facilities shapefile
   - Must contain geometry information for proximity calculations

3. **`candidates_new.shp`** (for appendix generation)
   - Additional candidate information including county data
   - Required columns: `FID`, `ComplexCom`, county name column

## Configuration

### Key Parameters
```python
MAX_BUDGET = 175e6  # Maximum budget ($175 million)
COST_NO_SERVICE_MIN = 800000  # Minimum no-service cost
COST_NO_SERVICE_MAX = 1.1e6   # Maximum no-service cost
COST_FULL_SERVICE_MIN = 11e6  # Minimum full-service cost
COST_FULL_SERVICE_MAX = 14e6  # Maximum full-service cost
```

### Excluded Facility Types
The system automatically filters out inappropriate facilities:
- DMV
- License
- Welcome Center
- Municipal
- Courthouse
- Rest Area

### Scoring Weights
```python
WEIGHTS = {
    'traffic_influx': 0.25,
    'crash_risk': 0.25,
    'accessibility': 0.25,
    'capacity': 0.25
}
```

## Usage

### Basic Execution
```bash
python iterate.py
```

### Output Structure
```
results/
├── iterative_results/
│   ├── no_service_continuous/
│   └── full_service_continuous/
├── selection_order/
│   ├── no_service_selection_order.csv
│   ├── full_service_selection_order.csv
│   ├── no_service_selection_order_readable.csv
│   └── full_service_selection_order_readable.csv
├── continuous_selection/
│   ├── no_service_continuous_selection.csv
│   ├── full_service_continuous_selection.csv
│   └── budget_summary_statistics.csv
├── iterations/
│   ├── no_service/
│   │   ├── iteration_001.csv
│   │   ├── iteration_002.csv
│   │   └── ...
│   └── full_service/
│       ├── iteration_001.csv
│       ├── iteration_002.csv
│       └── ...
├── figures/
│   ├── continuous_step_plot.png
│   └── continuous_step_plot.pdf
└── appendices/
    ├── Appendix_NO_SERVICE_Selected_Facilities.csv
    └── Appendix_FULL_SERVICE_Selected_Facilities.csv
```

## Key Functions

### Core Functions

#### `load_data()`
- Loads candidate locations and existing facilities
- Removes duplicates and filters inappropriate facility types
- Calculates development costs and proximity metrics

#### `infinite_budget_allocation()`
- Main optimization engine
- Iteratively selects facilities until budget is exhausted
- Updates proximity scores after each selection
- Tracks cumulative budget and capacity

#### `calculate_proximity_to_existing()`
- Calculates distance to nearest existing facilities
- Different strategies for no-service vs. full-service facilities
- Normalizes proximity scores for composite scoring

#### `calculate_development_costs()`
- Dynamic cost calculation based on facility capacity
- Separate models for no-service and full-service facilities

### Analysis Functions

#### `create_step_plot()`
- Generates staircase plots showing facility count vs. budget
- Compares no-service and full-service scenarios

#### `save_complete_facility_tables()`
- Exports detailed facility selection results
- Includes cumulative budget and capacity tracking

#### `create_appendix_tables()`
- Generates summary tables for different budget levels
- Cross-references with county and interstate information

## Cost Models

### No-Service Facilities
```
Cost = $200,000 (site prep) + ($10,000 × capacity)
```

### Full-Service Facilities
```
Cost = $3,000,000 (site prep) + ($100,000 × capacity)
```

## Scoring Methodology

### Composite Score Calculation
The system uses a weighted composite score including:
- **Traffic Influx** (25%): Normalized traffic volume data
- **Crash Risk** (25%): Safety considerations
- **Accessibility** (25%): Ease of access for users
- **Capacity** (25%): Facility size and accommodation capability
- **Proximity** (40%): Strategic placement relative to existing facilities

### Proximity Strategy
- **No-Service**: Prefers locations closer to existing facilities
- **Full-Service**: Prefers locations farther from existing facilities

## Output Files

### Selection Order Files
- **`*_selection_order.csv`**: Raw selection data with full precision
- **`*_selection_order_readable.csv`**: Human-readable format with costs in millions

### Summary Files
- **`budget_summary_statistics.csv`**: Facility counts and capacities at various budget levels
- **`continuous_step_plot.png/pdf`**: Visual comparison of allocation strategies

### Appendix Files
- Cross-tabulated facility selections by budget level
- Includes county and interstate information for selected facilities

## Technical Notes

### Coordinate Systems
- Uses EPSG:4326 (WGS84) coordinate reference system
- Ensures consistent CRS across all spatial datasets

### Duplicate Handling
- Removes duplicates based on `ComplexNam` field
- Filters facilities with missing or invalid names

### Memory Management
- Creates working copies of data for iterative processing
- Preserves original datasets for multiple scenario runs

## Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure all required input files are in the correct directories
2. **CRS Mismatches**: The system automatically reprojects data to consistent CRS
3. **Memory Issues**: Large datasets may require increased system memory
4. **Column Name Variations**: Check county column names in `candidates_new.shp`

### Error Handling
The system includes robust error handling for:
- Missing or corrupted input files
- Inconsistent data formats
- Geometric calculation errors
- Budget constraint violations

## Contributing

When modifying the code:
1. Maintain the existing parameter structure
2. Ensure all output files follow the established naming conventions
3. Test with both no-service and full-service scenarios
4. Verify coordinate system consistency

