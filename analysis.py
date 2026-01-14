import pandas as pd  # A tool for working with tables of data (like Excel for Python)
import yfinance as yf  # A tool to download stock market data from Yahoo Finance
import numpy as np  # A tool for advanced math calculations
import os  # A tool for interacting with the computer's operating system

# -----------------------------------------------------------------------------
# 1. LIST OF FUNDS (TICKERS)
# -----------------------------------------------------------------------------
# These are the codes used to identify specific funds on the stock market.
# We are looking at "UK Equity" funds (funds that invest in UK companies).
tickers = [
    'ISF.L',   # iShares Core FTSE 100 ETF (Top 100 biggest UK companies)
    'VMID.L',  # Vanguard FTSE 250 ETF (Next 250 medium-sized UK companies)
    'VUKG.L',  # Vanguard FTSE 100 ETF (Reinvests dividends automatically)
    'CUKX.L',  # iShares FTSE 100 ETF (Another fund tracking the top 100)
    'XUKX.L',  # Xtrackers FTSE 100 ETF (Tracks the same top 100 index)
    'IUKD.L',  # iShares UK Dividend ETF (Focuses on companies paying high dividends)
    'VUKE.L',  # Vanguard FTSE 100 ETF (Distributes dividends to you)
    'XUKS.L',  # Xtrackers FTSE 250 ETF (Tracks medium-sized companies)
    'UKRE.L'   # iShares UK Property ETF (Invests in UK real estate companies)
]

# -----------------------------------------------------------------------------
# 2. FETCHING THE DATA
# -----------------------------------------------------------------------------
def fetch_data(tickers, start='2019-01-01', end='2025-01-01'):
    """
    Downloads historical price data for the funds listed above.
    """
    print(f"Fetching data for {len(tickers)} tickers...")
    
    all_series = {} # A dictionary to store data for each fund temporarily
    
    for t in tickers:
        try:
            # yfinance downloads data from the internet
            # 'interval=1mo' means we want one price point per month
            df = yf.download(t, start=start, end=end, interval='1mo', progress=False)
            
            if not df.empty:
                # We need the 'Adj Close' price. This is the closing price adjusted 
                # for dividends and stock splits, giving a truer picture of return.
                
                # Check if the data has a complex structure (MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    if 'Adj Close' in df.columns.levels[0]:
                        series = df['Adj Close'].iloc[:, 0]
                    else:
                        series = df['Close'].iloc[:, 0]
                else:
                    # Simple structure
                    if 'Adj Close' in df.columns:
                        series = df['Adj Close']
                    else:
                        series = df['Close']
                
                # Make sure we just have a simple list of numbers (Series)
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                
                all_series[t] = series
        except Exception as e:
            print(f"Failed to download {t}: {e}")
            
    if not all_series:
        print("No data could be downloaded for any tickers.")
        return pd.DataFrame()
        
    # Combine all individual lists into one big table (DataFrame)
    prices = pd.DataFrame(all_series)
    
    print(f"Prices shape: {prices.shape}") # Prints (Rows, Columns)
    print(f"Columns: {prices.columns.tolist()}")
    return prices

# -----------------------------------------------------------------------------
# 3. PROCESSING THE ANALYSIS
# -----------------------------------------------------------------------------
def process_analysis(prices):
    """
    Calculates returns and ranks funds into groups (tertiles).
    """
    # Calculate 'Monthly Return': How much did the price go up or down this month?
    # pct_change() calculates the percentage change between the current and prior element.
    # fill_method=None prevents errors in newer pandas versions.
    returns = prices.pct_change(fill_method=None).dropna(how='all')
    
    # Calculate '12-Month Trailing Return': How much did the price change over the last year?
    # This helps us identify "Winners" (did well last year) vs "Losers".
    trailing_12m = prices.pct_change(12, fill_method=None).dropna(how='all')
    
    print(f"Returns shape: {returns.shape}")
    print(f"Trailing 12m shape: {trailing_12m.shape}")
    
    analysis_df = [] # A list to store our results month by month
    
    # Loop through each month in our data...
    for date in trailing_12m.index[:-1]: # Stop one month early because we need to see the *future* return
        try:
            # Find the date of the very next month
            next_date = returns.index[returns.index > date][0]
        except IndexError:
            continue # If there is no next month, skip this step
        
        # Get the past year's performance for all funds at this specific date
        current_trailing = trailing_12m.loc[date]
        
        # Get the performance for the VERY NEXT month (to see if the trend continues)
        next_month_ret = returns.loc[next_date]
        
        # Create a small table for just this month
        temp_df = pd.DataFrame({
            'trailing_12m': current_trailing, # Past Performance
            'next_month_ret': next_month_ret  # Future Performance
        }).dropna()
        
        if len(temp_df) < 3: 
            continue # We need at least 3 funds to divide them into 3 groups (Tertiles)
            
        # ---------------------------------------------------------------------
        # KEY STEP: RANKING
        # We split the funds into 3 equal groups (Tertiles) based on past performance.
        # Group 3 = Top 33% (Winners)
        # Group 2 = Middle 33%
        # Group 1 = Bottom 33% (Losers)
        # ---------------------------------------------------------------------
        num_buckets = 3 
        try:
            # qcut divides the data into equal-sized buckets (quantiles)
            temp_df['quintile'] = pd.qcut(temp_df['trailing_12m'], num_buckets, labels=False, duplicates='drop') + 1
            temp_df['date'] = date
            temp_df['ticker'] = temp_df.index
            analysis_df.append(temp_df)
        except ValueError as e:
            print(f"Error at {date}: {e}")
            continue
    
    if not analysis_df:
        print("Warning: analysis_df is empty. No valid months found for analysis.")
        return pd.DataFrame()
        
    # Combine all the monthly tables into one giant table
    full_analysis = pd.concat(analysis_df)
    return full_analysis

# -----------------------------------------------------------------------------
# 4. COMPUTE TRANSITIONS
# -----------------------------------------------------------------------------
def compute_transitions(df):
    """
    Calculates how often a fund moves from one group to another.
    Example: Does a 'Winner' (Group 3) stay a 'Winner' in the next period?
    """
    # Sort data by Fund Name and then Date to ensure time flows correctly
    df = df.sort_values(['ticker', 'date'])
    
    # Create a new column 'next_quintile'
    # .shift(-1) looks at the NEXT row's rank. 
    # This lets us compare "Rank Today" vs "Rank Tomorrow"
    df['next_quintile'] = df.groupby('ticker')['quintile'].shift(-1)
    
    # Remove rows where we don't know the future rank (the last month of data)
    transition_df = df.dropna(subset=['next_quintile'])
    
    return transition_df

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Get the data
    prices = fetch_data(tickers)
    
    if prices.empty:
        print("No data fetched. Check internet connection or tickers.")
    else:
        # Save raw price data
        prices.to_csv('monthly_prices.csv')
        
        # Step 2: Analyze performance and rank funds
        analysis = process_analysis(prices)
        
        if analysis.empty:
            print("Analysis could not be performed due to insufficient data.")
        else:
            # Save the detailed analysis
            analysis.to_csv('monthly_analysis.csv', index=False)
            
            # Step 3: Calculate transitions (changes in rank over time)
            transitions = compute_transitions(analysis)
            transitions.to_csv('quintile_transitions.csv', index=False)
            
            # Step 4: Create a summary for simple charts
            # Calculate the average return for each group
            summary = analysis.groupby('quintile')['next_month_ret'].mean().reset_index()
            summary.to_csv('quintile_performance_summary.csv', index=False)
            
            print("Analysis complete. Files generated:")
            print("- monthly_prices.csv (Raw Data)")
            print("- monthly_analysis.csv (Detailed Calculations)")
            print("- quintile_transitions.csv (For Heatmaps)")
            print("- quintile_performance_summary.csv (For Bar Charts)")