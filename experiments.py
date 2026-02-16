import pandas as pd
import os
import time
from tqdm import tqdm
from train_vectors import Config, Pipeline
from metrics import NetworkMetrics

# Constants
EXPERIMENTS_FILE = 'experiments.csv'
RESULTS_FILE = 'reports/experiments_results.csv'
PROBE_TICKERS = ['AAPL', 'JPM', 'JNJ', 'XOM', 'TSLA', 'AMD', 'LOW', 'KO', 'PEP', 'AMZN']
SECTOR_FILE = 'data/Holdings_details_Russell_3000_ETF.csv'

def load_sector_map():
    """Parses the Russell 3000 ETF holdings file to map Ticker -> Sub-Industry."""
    print(f"📂 Loading sector map from {SECTOR_FILE}...")
    try:
        # File has a messy header, actual data starts around line 9 (header on line 8)
        # We'll rely on pandas to find the header row containing 'TICKER'
        df = pd.read_csv(SECTOR_FILE, header=7) # Line 8 is header (0-indexed 7)
        
        # Verify columns exist
        if 'TICKER' not in df.columns or 'SUB-INDUSTRY' not in df.columns:
            print("   ⚠️  Columns TICKER or SUB-INDUSTRY not found. Trying auto-detect...")
            df = pd.read_csv(SECTOR_FILE, skiprows=7)
        
        # Clean and create map
        df = df.dropna(subset=['TICKER', 'SUB-INDUSTRY'])
        sector_map = dict(zip(df['TICKER'], df['SUB-INDUSTRY']))
        print(f"   ℹ️  Loaded {len(sector_map)} ticker classifications.")
        return sector_map
    except Exception as e:
        print(f"   ❌ Failed to load sector map: {e}")
        return {}

def run_all_experiments():
    # 0. Load Sector Map
    sector_map = load_sector_map()

    # 1. Load Experiments
    if not os.path.exists(EXPERIMENTS_FILE):
        print(f"❌ Error: {EXPERIMENTS_FILE} not found.")
        return

    experiments_df = pd.read_csv(EXPERIMENTS_FILE)
    print(f"🚀 Found {len(experiments_df)} experiments to run.")

    # 📂 Load existing results to append to them
    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            results = pd.read_csv(RESULTS_FILE).to_dict('records')
            print(f"   🔄 Loaded {len(results)} existing results from {RESULTS_FILE}")
        except Exception as e:
            print(f"   ⚠️ Could not read existing file: {e}")

    # 2. Iterate through experiments
    existing_ids = set(r['run_id'] for r in results)  # Cache existing IDs
    
    for idx, row in experiments_df.iterrows():
        run_id = row['run_id']
        
        if run_id in existing_ids:
            print(f"⏩ Skipping Run {run_id}: Already found in results.")
            continue

        universe = row['universe']
        period = row['period']
        print(f"\n🧪 Running Experiment {run_id}: {universe} / {period} / {row['hypothesis']}")

        # Construct cache file path based on universe and period
        # Mapping: sp500 -> data/sp500_{period}.csv
        #          r1000 -> data/russell_1000_{period}.csv
        #          r3000 -> data/russell_3000_{period}.csv
        
        file_prefix = "sp500" if universe == "sp500" else (
            "russell_1000" if universe == "r1000" else "russell_3000"
        )
        cache_file = f"data/{file_prefix}_{period}.csv"

        if not os.path.exists(cache_file):
            print(f"   ⚠️  Skipping: Data file {cache_file} not found.")
            row['status'] = 'Skipped (File Not Found)'
            results.append(row.to_dict())
            continue

        # 3. Configure Training
        config = Config(training_period=period)
        config.cache_file = cache_file
        config.walk_p = float(row['walk_p'])
        config.walk_q = float(row['walk_q'])
        config.dimensions = int(row['dimensions'])
        config.correlation_threshold = float(row['threshold'])
        config.use_precision_matrix = bool(row['use_lasso'])
        config.graph_filtering = str(row.get('graph_mode', 'threshold')).strip() # Default to threshold
        config.skip_pinecone = True # Run locally
        config.include_metadata = False 
        
        # Override index name just in case, though unused
        config.index_name = f"exp_{run_id}"

        try:
            # 4. Run Pipeline
            pipeline = Pipeline(config)
            graph, embeddings, returns = pipeline.run_experiment()

            # 5. Run Metrics
            metrics_engine = NetworkMetrics(graph, embeddings, sector_map)
            stats = metrics_engine.calculate_metrics()
            
            # 6. Semantic Sampling
            semantic_str = metrics_engine.get_semantic_samples(PROBE_TICKERS)

            # 7. Aggregate Results
            result_row = row.to_dict()
            result_row.update(stats)
            result_row['semantic_samples'] = semantic_str
            result_row['status'] = 'Success'
            result_row['num_tickers'] = len(embeddings)
            
            results.append(result_row)
            print(f"   ✅ OK. Modularity: {stats.get('modularity', 0):.3f} | Gini: {stats.get('gini_score', 0):.3f}")

            # 💾 Save incrementally so we don't lose progress
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            row['status'] = f'Failed: {str(e)}'
            results.append(row.to_dict())
            # Save even on failure
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    # 8. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n🎉 Done! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_all_experiments()
