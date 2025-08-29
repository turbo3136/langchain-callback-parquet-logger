"""
Simple before/after comparison showing how batch_run simplifies batch processing.
"""

import asyncio
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_callback_parquet_logger import ParquetLogger, with_tags, batch_run


# ============================================================================
# BEFORE: Manual approach with RunnableLambda
# ============================================================================

async def manual_approach():
    """The old way - manual setup with RunnableLambda."""
    print("="*60)
    print("BEFORE: Manual RunnableLambda approach")
    print("="*60 + "\n")
    
    # Sample data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'question': ['What is 2+2?', 'What is the capital of France?', 'Who wrote Hamlet?']
    })
    
    with ParquetLogger("./logs") as logger:
        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
        
        # Manual progress tracking
        completed = 0
        total = len(df)
        
        async def process_row(row):
            nonlocal completed
            result = await llm.ainvoke(
                input=row['question'],
                config=with_tags(custom_id=str(row['id']))
            )
            completed += 1
            print(f"Progress: {completed}/{total} ({completed*100//total}%)")
            return result
        
        # Manual runner setup
        runner = RunnableLambda(process_row)
        results = await runner.abatch(
            df.to_dict('records'),
            config={"max_concurrency": 2},
            return_exceptions=True
        )
        
        # Display results
        for i, (q, r) in enumerate(zip(df['question'], results)):
            print(f"\nQ{i+1}: {q}")
            print(f"A{i+1}: {r.content if hasattr(r, 'content') else r}")
    
    return results


# ============================================================================
# AFTER: Simplified approach with batch_run
# ============================================================================

async def simplified_approach():
    """The new way - using batch_run helper."""
    print("\n" + "="*60)
    print("AFTER: Simplified batch_run approach")
    print("="*60 + "\n")
    
    # Same sample data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'question': ['What is 2+2?', 'What is the capital of France?', 'Who wrote Hamlet?']
    })
    
    # Prepare DataFrame columns
    df['prompt'] = df['question']  # Can transform if needed
    df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))
    
    with ParquetLogger("./logs") as logger:
        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
        
        # Single line replaces all the manual setup!
        results = await batch_run(
            df=df,
            llm=llm,
            max_concurrency=2,
            show_progress=True
        )
        
        # Display results
        df['answer'] = results
        for _, row in df.iterrows():
            print(f"\nQ{row['id']}: {row['question']}")
            answer = row['answer']
            print(f"A{row['id']}: {answer.content if hasattr(answer, 'content') else answer}")
    
    return results


# ============================================================================
# COMPARISON: Show the difference
# ============================================================================

async def main():
    """Run both approaches to show the difference."""
    print("\n🔍 BATCH PROCESSING COMPARISON\n")
    
    # Run manual approach
    manual_results = await manual_approach()
    
    # Run simplified approach
    simplified_results = await simplified_approach()
    
    print("\n" + "="*60)
    print("✅ SUMMARY")
    print("="*60)
    print("""
Manual approach requires:
- Custom progress tracking implementation
- Manual runner setup with RunnableLambda
- Manual async function definition
- ~20 lines of boilerplate code

Simplified batch_run approach:
- Automatic progress bar (notebook-aware)
- Single function call
- Handles all async batching
- 3 lines of code

The simplified approach is:
✓ Cleaner and more readable
✓ Less error-prone
✓ Automatically handles progress
✓ Same results with less code
""")


if __name__ == "__main__":
    asyncio.run(main())