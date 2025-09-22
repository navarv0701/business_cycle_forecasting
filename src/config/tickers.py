"""
Bloomberg ticker definitions for business cycle forecasting
"""

BLOOMBERG_TICKERS = {
    # Growth indicators for ISM PMI model
    'growth_indicators': {
        'USGG10YR Index': {
            'name': 'US_10Y_Yield',
            'description': 'US 10-Year Treasury Yield',
            'frequency': 'daily'
        },
        'USGG2YR Index': {
            'name': 'US_2Y_Yield',
            'description': 'US 2-Year Treasury Yield',
            'frequency': 'daily'
        },
        'NHSPSTOT Index': {
            'name': 'Housing_Permits',
            'description': 'US Housing Permits Total',
            'frequency': 'monthly'
        },
        'CONSSENT Index': {
            'name': 'Consumer_Confidence',
            'description': 'Consumer Confidence Index',
            'frequency': 'monthly'
        },
        'CGNOXAI Index': {
            'name': 'Core_Capital_Goods',
            'description': 'Core Capital Goods New Orders',
            'frequency': 'monthly'
        },
        'CGNOXMOM Index': {
            'name': 'Core_Capital_Goods_MoM',
            'description': 'Core Capital Goods Month-over-Month',
            'frequency': 'monthly'
        },
        'INJCAVG Index': {
            'name': 'Unemployment_Claims_4WK',
            'description': 'Initial Jobless Claims 4-Week Average',
            'frequency': 'weekly'
        }
    },
    
    # Inflation indicators
    'inflation_indicators': {
        'NAPMPMI Index': {
            'name': 'ISM_PMI',
            'description': 'ISM Manufacturing PMI',
            'frequency': 'monthly'
        },
        'ECIWAG Index': {
            'name': 'Employment_Cost_Index',
            'description': 'Employment Cost Index - Wages',
            'frequency': 'quarterly'
        },
        'USGG5Y5Y Index': {
            'name': 'Inflation_Expect_5Y5Y',
            'description': '5-Year 5-Year Forward Inflation Expectation',
            'frequency': 'daily'
        },
        'USIMPY Index': {
            'name': 'Import_Price_Index',
            'description': 'US Import Price Index YoY',
            'frequency': 'monthly'
        },
        'BDIY Index': {
            'name': 'Baltic_Dry_Index',
            'description': 'Baltic Dry Index',
            'frequency': 'daily'
        },
        'NFICCMPP Index': {
            'name': 'NFIB_Comp_Plans',
            'description': 'NFIB Small Business Compensation Plans',
            'frequency': 'monthly'
        }
    },
    
    # Target variables
    'target_variables': {
        'USPPIY Index': {
            'name': 'PPI_YoY',
            'description': 'US Producer Price Index YoY',
            'frequency': 'monthly'
        },
        'USCPIY Index': {
            'name': 'CPI_YoY',
            'description': 'US Consumer Price Index YoY',
            'frequency': 'monthly'
        }
    }
}

# Flatten for easy access
ALL_TICKERS = {}
for category in BLOOMBERG_TICKERS.values():
    ALL_TICKERS.update(category)