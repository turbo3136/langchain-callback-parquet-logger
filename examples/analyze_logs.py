"""
Analyze logs with v1.0.0 standardized payload structure.
Shows how to extract insights from the new nested JSON format.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any


def analyze_logs(log_dir: str = "./logs") -> Dict[str, Any]:
    """
    Comprehensive log analysis using v1.0.0 payload structure.
    
    Returns dict with metrics, errors, and insights.
    """
    # Read all parquet files
    df = pd.read_parquet(log_dir)
    print(f"📊 Analyzing {len(df)} log entries from {log_dir}")
    print("=" * 60)
    
    # Parse v1.0.0 standardized payloads
    def parse_payload(row):
        """Extract key fields from v1.0.0 payload structure."""
        payload = json.loads(row['payload'])
        
        # Navigate the standardized structure
        return {
            'event_type': payload['event_type'],
            'event_phase': payload['event_phase'],
            'event_component': payload['event_component'],
            'custom_id': payload['execution']['custom_id'],
            'model': payload['data']['config']['model'],
            'prompts': payload['data']['inputs'].get('prompts', []),
            'response': payload['data']['outputs'].get('response', {}),
            'usage': payload['data']['outputs'].get('usage', {}),
            'error_message': payload['data']['error'].get('message', ''),
            'timestamp_iso': payload['timestamp']
        }
    
    # Extract structured data
    df['parsed'] = df.apply(parse_payload, axis=1)
    df_parsed = pd.json_normalize(df['parsed'])
    
    # 1. Event Type Distribution
    print("\n📈 Event Distribution:")
    event_counts = df['event_type'].value_counts()
    for event, count in event_counts.items():
        print(f"  {event}: {count}")
    
    # 2. Token Usage Analysis
    llm_ends = df_parsed[df_parsed['event_type'] == 'llm_end']
    if not llm_ends.empty:
        total_tokens = llm_ends['usage'].apply(
            lambda x: x.get('total_tokens', 0) if isinstance(x, dict) else 0
        ).sum()
        
        print(f"\n💰 Token Usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Average per call: {total_tokens / len(llm_ends):.1f}")
        
        # Estimate costs (GPT-4o-mini pricing as of 2024)
        cost = (total_tokens / 1000) * 0.00015  # $0.15 per 1M tokens
        print(f"  Estimated cost: ${cost:.4f}")
    
    # 3. Error Analysis
    errors = df_parsed[df_parsed['error_message'] != '']
    if not errors.empty:
        print(f"\n⚠️  Errors Found: {len(errors)}")
        for _, error in errors.iterrows():
            print(f"  - {error['custom_id']}: {error['error_message'][:50]}...")
    
    # 4. Performance Metrics
    print(f"\n⏱️  Performance:")
    
    # Calculate response times by matching start/end events
    starts = df[df['event_type'] == 'llm_start'][['run_id', 'timestamp']]
    ends = df[df['event_type'] == 'llm_end'][['run_id', 'timestamp']]
    
    if not starts.empty and not ends.empty:
        merged = starts.merge(ends, on='run_id', suffixes=('_start', '_end'))
        merged['duration_ms'] = (
            (merged['timestamp_end'] - merged['timestamp_start']).dt.total_seconds() * 1000
        )
        
        print(f"  Average response time: {merged['duration_ms'].mean():.0f}ms")
        print(f"  Min: {merged['duration_ms'].min():.0f}ms")
        print(f"  Max: {merged['duration_ms'].max():.0f}ms")
    
    # 5. Custom ID Tracking
    custom_ids = df_parsed[df_parsed['custom_id'] != '']
    if not custom_ids.empty:
        print(f"\n🏷️  Custom ID Summary:")
        print(f"  Total tracked requests: {custom_ids['custom_id'].nunique()}")
        
        # Show top custom IDs by frequency
        top_ids = custom_ids['custom_id'].value_counts().head(5)
        print("  Most frequent IDs:")
        for custom_id, count in top_ids.items():
            print(f"    {custom_id}: {count} events")
    
    # 6. Model Usage
    models = df_parsed[df_parsed['model'] != ''].groupby('model').size()
    if not models.empty:
        print(f"\n🤖 Models Used:")
        for model, count in models.items():
            print(f"  {model}: {count} calls")
    
    return {
        'total_events': len(df),
        'event_distribution': event_counts.to_dict(),
        'total_tokens': total_tokens if 'total_tokens' in locals() else 0,
        'error_count': len(errors),
        'unique_custom_ids': custom_ids['custom_id'].nunique() if not custom_ids.empty else 0
    }


def query_with_duckdb(log_dir: str = "./logs"):
    """
    Advanced queries using DuckDB with v1.0.0 payload paths.
    Requires: pip install duckdb
    """
    try:
        import duckdb
    except ImportError:
        print("⚠️  DuckDB not installed. Run: pip install duckdb")
        return
    
    print("\n" + "=" * 60)
    print("🦆 DuckDB Analysis (v1.0.0 Payload Structure)")
    print("=" * 60)
    
    conn = duckdb.connect()
    
    # Query 1: Token usage by custom ID
    query1 = """
    SELECT 
        logger_custom_id,
        COUNT(*) as call_count,
        SUM(CAST(json_extract_string(payload, '$.data.outputs.usage.total_tokens') AS INTEGER)) as total_tokens,
        AVG(CAST(json_extract_string(payload, '$.data.outputs.usage.total_tokens') AS INTEGER)) as avg_tokens
    FROM read_parquet(?)
    WHERE event_type = 'llm_end'
        AND logger_custom_id != ''
    GROUP BY logger_custom_id
    ORDER BY total_tokens DESC
    LIMIT 10
    """
    
    print("\n📊 Top 10 Users by Token Usage:")
    result = conn.execute(query1, [f"{log_dir}/**/*.parquet"]).df()
    print(result.to_string())
    
    # Query 2: Error rate by model
    query2 = """
    WITH model_calls AS (
        SELECT 
            json_extract_string(payload, '$.data.config.model') as model,
            event_type,
            CASE WHEN event_type = 'llm_error' THEN 1 ELSE 0 END as is_error
        FROM read_parquet(?)
        WHERE event_type IN ('llm_end', 'llm_error')
    )
    SELECT 
        model,
        COUNT(*) as total_calls,
        SUM(is_error) as errors,
        ROUND(100.0 * SUM(is_error) / COUNT(*), 2) as error_rate_pct
    FROM model_calls
    WHERE model != ''
    GROUP BY model
    ORDER BY total_calls DESC
    """
    
    print("\n⚠️  Error Rates by Model:")
    result = conn.execute(query2, [f"{log_dir}/**/*.parquet"]).df()
    print(result.to_string())
    
    # Query 3: Hourly usage pattern
    query3 = """
    SELECT 
        EXTRACT(HOUR FROM timestamp) as hour,
        COUNT(*) as requests,
        COUNT(DISTINCT logger_custom_id) as unique_users
    FROM read_parquet(?)
    WHERE event_type = 'llm_start'
    GROUP BY hour
    ORDER BY hour
    """
    
    print("\n🕐 Hourly Usage Pattern:")
    result = conn.execute(query3, [f"{log_dir}/**/*.parquet"]).df()
    if not result.empty:
        for _, row in result.iterrows():
            bar = '█' * int(row['requests'] / result['requests'].max() * 20)
            print(f"  {int(row['hour']):02d}:00 {bar} {int(row['requests'])} requests")
    
    conn.close()


def main():
    """Run all analysis examples."""
    
    # First, create some sample logs
    print("Creating sample logs for analysis...")
    from langchain_callback_parquet_logger import ParquetLogger, with_tags
    from langchain_openai import ChatOpenAI
    
    with ParquetLogger("./sample_logs") as logger:
        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
        
        # Generate some varied logs
        for i in range(5):
            try:
                response = llm.invoke(
                    f"Question {i}: What is {i}+{i}?",
                    config=with_tags(custom_id=f"demo-user-{i % 2}")
                )
            except Exception:
                pass  # Generate some errors for demo
    
    print("\n" + "=" * 60)
    
    # Run analysis
    metrics = analyze_logs("./sample_logs")
    
    # Try DuckDB analysis if available
    query_with_duckdb("./sample_logs")
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print(f"   Processed {metrics['total_events']} events")


if __name__ == "__main__":
    main()