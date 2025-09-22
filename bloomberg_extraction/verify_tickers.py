#!/usr/bin/env python3
"""
Bloomberg Ticker Verification Script
Run this on Bloomberg Terminal to verify all tickers before extraction
"""

import sys
try:
    import blpapi
except ImportError:
    print("Error: blpapi not installed. Run: pip install blpapi")
    sys.exit(1)

def verify_tickers():
    """Test each ticker to see if it returns data"""
    
    # List of tickers to verify - UPDATE THESE
    TICKERS_TO_TEST = {
        # Yields
        'USGG10YR Index': 'US 10-Year Yield',
        'USGG2YR Index': 'US 2-Year Yield',
        'USGG5Y5Y Index': '5Y5Y Inflation Expectations',
        
        # Economic Indicators - THESE LIKELY NEED FIXES
        'CONSSENT Index': 'Consumer Confidence (probably wrong)',
        'CONCCONF Index': 'Conference Board Confidence (try this)',
        'NAPMPMI Index': 'ISM PMI (probably wrong)',
        'ISM MANU Index': 'ISM Manufacturing (try this)',
        'NAPMPMI Index': 'ISM Manufacturing PMI (old ticker)',
        
        # Housing
        'NHSPSTOT Index': 'Housing Permits',
        'PERMIT Index': 'Building Permits (alternative)',
        'NHSPST Index': 'Housing Starts (alternative)',
        
        # Employment
        'INJCAVG Index': 'Claims 4-week average (probably wrong)',
        'INJCJC4W Index': 'Initial Claims 4-Week MA (try this)',
        'INJCJC Index': 'Initial Jobless Claims',
        'ECIWAG Index': 'Employment Cost Index',
        
        # Inflation
        'USCPIY Index': 'CPI YoY (probably wrong)',
        'USPPIY Index': 'PPI YoY (probably wrong)',
        'CPI YOY Index': 'CPI Year-over-Year (try this)',
        'CPI XYOY Index': 'CPI YoY (alternative)',
        'PPI YOY Index': 'PPI Year-over-Year (try this)',
        'PPI XYOY Index': 'PPI YoY (alternative)',
        
        # Trade/Commodities
        'BDIY Index': 'Baltic Dry Index',
        'USIMPY Index': 'Import Price Index (probably wrong)',
        'IMPORT Index': 'Import Prices (alternative)',
        
        # Business
        'CGNOXAI Index': 'Core Capital Goods Orders',
        'DGNOXTCH Index': 'Durable Goods ex-Transport (alternative)',
        'CGNOXMOM Index': 'Core Capital Goods MoM',
        'NFICCMPP Index': 'NFIB Compensation Plans',
        'NFIB SMLB Index': 'NFIB Small Business (alternative)',
    }
    
    print("="*60)
    print("BLOOMBERG TICKER VERIFICATION")
    print("="*60)
    
    # Connect to Bloomberg
    try:
        session = blpapi.Session()
        if not session.start():
            print("Failed to start Bloomberg session")
            return
        if not session.openService("//blp/refdata"):
            print("Failed to open refdata service")
            return
        service = session.getService("//blp/refdata")
    except Exception as e:
        print(f"Connection error: {e}")
        return
    
    # Test each ticker
    working_tickers = []
    failed_tickers = []
    
    for ticker, description in TICKERS_TO_TEST.items():
        try:
            # Create request for reference data
            request = service.createRequest("ReferenceDataRequest")
            request.append("securities", ticker)
            request.append("fields", "PX_LAST")  # Just get last price
            
            # Send request
            session.sendRequest(request)
            
            # Check response
            has_data = False
            while True:
                event = session.nextEvent(500)
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        secData = msg.getElement("securityData")
                        
                        # Check for errors
                        if secData.hasElement("securityError"):
                            error = secData.getElement("securityError")
                            print(f"✗ {ticker:<20} - ERROR: {error.getElementAsString('message')}")
                            failed_tickers.append(ticker)
                        else:
                            # Check if we got data
                            if secData.hasElement("fieldData"):
                                fieldData = secData.getElement("fieldData")
                                if fieldData.hasElement("PX_LAST"):
                                    value = fieldData.getElementAsFloat("PX_LAST")
                                    print(f"✓ {ticker:<20} - OK (Last: {value:.2f})")
                                    working_tickers.append(ticker)
                                    has_data = True
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if not has_data and ticker not in failed_tickers:
                print(f"? {ticker:<20} - No data returned")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"✗ {ticker:<20} - Exception: {e}")
            failed_tickers.append(ticker)
    
    # Disconnect
    session.stop()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Working tickers: {len(working_tickers)}")
    print(f"✗ Failed tickers: {len(failed_tickers)}")
    
    if working_tickers:
        print("\nWORKING TICKERS TO USE:")
        print("-"*40)
        for ticker in working_tickers:
            print(f"  '{ticker}'")
    
    if failed_tickers:
        print("\nFAILED TICKERS TO REPLACE:")
        print("-"*40)
        for ticker in failed_tickers:
            print(f"  '{ticker}'")
    
    # Save results
    with open('ticker_verification_results.txt', 'w') as f:
        f.write("WORKING TICKERS:\n")
        for t in working_tickers:
            f.write(f"{t}\n")
        f.write("\nFAILED TICKERS:\n")
        for t in failed_tickers:
            f.write(f"{t}\n")
    
    print("\nResults saved to: ticker_verification_results.txt")
    print("\nNEXT STEPS:")
    print("1. Replace failed tickers in extract_data.py")
    print("2. Use Bloomberg Terminal to find correct ticker names")
    print("3. Type SECF <GO> on Terminal to search for securities")

if __name__ == "__main__":
    verify_tickers()