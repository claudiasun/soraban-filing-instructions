#!/usr/bin/env python3
"""
Example usage of LLM judge for OCR validation.

This demonstrates how to:
1. Validate a parquet file with OCR results
2. Analyze the validation results
3. Display documents needing review
"""

import os
import sys
from llm_judge import validate_parquet

def main():
    # Example 1: Validate a single file
    print("="*80)
    print("Example 1: Validating OCR results with LLM judge")
    print("="*80)
    
    input_file = "CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet"
    output_file = "CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_validated.parquet"
    pdf_dir = "Filing instructions"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please run the OCR script first:")
        print("  python run.py 'Filing instructions/CCH Axcess - Fed, CA, SC, PA & NC_Redacted.pdf'")
        return
    
    # Validate using LLM (you can change provider to 'anthropic' if preferred)
    provider = "anthropic"  # or "openai"
    
    print(f"\nValidating with {provider}...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"PDF directory: {pdf_dir}")
    
    try:
        validate_parquet(
            input_path=input_file,
            output_path=output_file,
            provider=provider,
            pdf_directory=pdf_dir
        )
        
        # Analyze results
        print("\n" + "="*80)
        print("Analyzing validation results...")
        print("="*80)
        
        try:
            import pandas as pd
            df = pd.read_parquet(output_file)
            
            print(f"\nTotal documents: {len(df)}")
            print(f"Aligned: {df['is_aligned'].sum()} / {len(df)}")
            print(f"Reasonable: {df['is_reasonable'].sum()} / {len(df)}")
            print(f"Average confidence: {df['confidence'].mean():.2f}")
            
            # Documents needing review
            needs_review = df[df['confidence'] < 0.7]
            print(f"\nDocuments needing human review (confidence < 0.7): {len(needs_review)}")
            
            if len(needs_review) > 0:
                print("\nDetails of documents needing review:")
                print("-" * 80)
                for idx, row in needs_review.iterrows():
                    print(f"\nDocument: {row['pdf_url']}")
                    print(f"  Extracted by OCR: {row['money_amounts']}")
                    print(f"  Found by LLM: {row['amounts_found_by_llm']}")
                    print(f"  Missing: {row['missing_amounts']}")
                    print(f"  Incorrect: {row['incorrect_amounts']}")
                    print(f"  Confidence: {row['confidence']:.2f}")
                    print(f"  Note: {row['note']}")
            
            # Misaligned extractions
            misaligned = df[~df['is_aligned']]
            if len(misaligned) > 0:
                print(f"\n{'='*80}")
                print(f"Misaligned extractions: {len(misaligned)}")
                print("="*80)
                for idx, row in misaligned.iterrows():
                    print(f"\nDocument: {row['pdf_url']}")
                    print(f"  Extracted: {row['money_amounts']}")
                    print(f"  LLM found: {row['amounts_found_by_llm']}")
                    print(f"  Note: {row['note']}")
            
        except ImportError:
            print("Warning: pandas not installed, skipping analysis")
            print("Install with: pip install pandas")
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        print("\nMake sure you have:")
        print("1. Set your API key:")
        if provider == "openai":
            print("   export OPENAI_API_KEY='sk-...'")
        else:
            print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print("2. Installed required packages:")
        print(f"   pip install {provider}")
        print("   pip install pandas pyarrow pdf2image pillow")
        return 1
    
    print("\n" + "="*80)
    print("✓ Validation complete!")
    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

