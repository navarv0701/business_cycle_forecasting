# Business Cycle Forecasting System

An econometric forecasting system for predicting economic growth (ISM PMI) and inflation (PPI/CPI) to support business cycle-based investment decisions.

## Overview

This system provides 6-9 month forecasts for key economic indicators using:
- **Growth Model**: VAR and factor models to forecast ISM PMI
- **PPI Model**: ARDL model for early inflation warning (3-6 month lead)
- **CPI Model**: ARDL model for inflation confirmation (6-9 month horizon)
- **Business Cycle Framework**: Regime identification and investment signals

## Installation

### On Development Machine (Laptop)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/business-cycle-forecasting.git
cd business-cycle-forecasting
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### On Bloomberg Terminal

1. Clone the repository:
```bash
git clone https://github.com/yourusername/business-cycle-forecasting.git
cd business-cycle-forecasting
```

2. Install minimal dependencies:
```bash
pip install -r requirements_bloomberg.txt
```

## Usage

### Step 1: Extract Bloomberg Data (Terminal)

Run on Bloomberg Terminal machine:

```bash
cd bloomberg_extraction
python extract_data.py

# For recent data only (last 2 years):
python extract_data.py --recent
```

This creates a `bloomberg_data_export` folder with all the data files.

### Step 2: Transfer Data

Copy the `bloomberg_data_export` folder to your development machine and place it in:
```
business-cycle-forecasting/data/bloomberg_cache/
```

### Step 3: Run Forecasting Models (Laptop)

```bash
# Using real Bloomberg data
python main.py --mode production

# Using sample data for development
python main.py --mode development --use-sample-data

# Generate detailed report
python main.py --generate-report --output outputs/reports
```

## Project Structure

```
business-cycle-forecasting/
├── bloomberg_extraction/    # Bloomberg data extraction (Terminal only)
├── src/                    # Source code
│   ├── config/            # Configuration and parameters
│   ├── data/              # Data loading and transformation
│   ├── models/            # Forecasting models
│   ├── utils/             # Utilities and visualization
│   └── analysis/          # Analysis and reporting
├── data/                   # Data storage
│   ├── bloomberg_cache/   # Bloomberg extracted data
│   └── sample_data/       # Sample data for development
├── outputs/               # Results and reports
└── notebooks/             # Jupyter notebooks for exploration
```

## Models

### Growth Model (ISM PMI)
- **Method**: Vector Autoregression (VAR) + Dynamic Factor Model
- **Features**: Yield spread, housing permits, consumer confidence, capital goods, unemployment claims
- **Horizons**: 3, 6, 9 months
- **Output**: ISM PMI forecast (expansion > 50, contraction < 50)

### PPI Model (Early Warning)
- **Method**: Autoregressive Distributed Lag (ARDL)
- **Features**: Baltic Dry Index (2-3 month lead), import prices, employment costs
- **Horizons**: 3, 6 months
- **Purpose**: Early detection of inflation turning points

### CPI Model (Confirmation)
- **Method**: ARDL with regime adjustments
- **Features**: Employment costs, NFIB compensation plans, inflation expectations
- **Horizons**: 6, 9 months  
- **Purpose**: Confirm inflation trends for policy implications

### Business Cycle Regimes
1. **Goldilocks**: Rising growth, stable inflation → Overweight equities
2. **Reflation**: Rising growth, rising inflation → Cyclical tilt
3. **Stagflation**: Falling growth, rising inflation → Defensive positioning
4. **Deflation**: Falling growth, falling inflation → Risk-off

## Output Files

After running the system, you'll find:

```
outputs/
├── forecasts/
│   └── forecasts.json         # Latest forecast values
├── reports/
│   ├── forecast_report_YYYYMMDD.md   # Detailed markdown report
│   └── forecast_data_YYYYMMDD.json   # Structured forecast data
└── charts/
    ├── forecast_charts.png    # Time series visualizations
    └── regime_analysis.png    # Business cycle analysis
```

## Development Workflow

### For Model Development (No Bloomberg Required)

1. Generate sample data:
```bash
python -c "from src.data.sample_generator import SampleDataGenerator; SampleDataGenerator().generate_all_data()"
```

2. Run with sample data:
```bash
python main.py --mode development
```

3. Experiment in Jupyter:
```bash
jupyter lab
# Open notebooks/01_data_exploration.ipynb
```

### For Production Updates

1. Monthly data refresh on Terminal:
```bash
cd bloomberg_extraction
python extract_data.py --recent
```

2. Update repository:
```bash
git add data/bloomberg_cache/
git commit -m "Update Bloomberg data"
git push
```

3. Run updated forecasts:
```bash
git pull
python main.py --mode production --generate-report
```

## Model Performance

Expected performance metrics (based on backtesting):

| Model | Directional Accuracy | RMSE | Lead Time |
|-------|---------------------|------|-----------|
| Growth (ISM PMI) | ~65% | 2.5 points | 6-9 months |
| PPI | ~60% | 1.0% | 3-6 months |
| CPI | ~70% | 0.5% | 6-9 months |
| Regime | ~65% | - | 6 months |

## Configuration

Key parameters can be adjusted in `src/config/model_params.py`:

- Forecast horizons
- Model lag orders
- Confidence intervals
- Regime thresholds
- Backtest windows

## Testing

Run tests with:
```bash
pytest tests/
```

## Troubleshooting

### Bloomberg Connection Issues
- Ensure Terminal is logged in
- Check firewall settings for port 8194
- Verify Bloomberg API installation

### Missing Data
- Check metadata.json for extraction details
- Verify all tickers have permissions
- Use --recent flag for quick updates

### Model Errors
- Ensure sufficient historical data (minimum 2 years)
- Check for missing required columns
- Review data quality in sample vs production

## Citation

If using this system for research or analysis:

```
Business Cycle Forecasting System
https://github.com/yourusername/business-cycle-forecasting
Forecasts ISM PMI, PPI, and CPI using econometric models
```

## License

This project is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## Contact

For questions or issues, please open a GitHub issue.

---

**Disclaimer**: This system generates economic forecasts for educational purposes. Investment decisions should incorporate additional analysis and professional advice.