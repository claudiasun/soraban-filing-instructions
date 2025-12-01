#!/usr/bin/env python3
"""
Analyze validation results from LLM judge.

This script helps you quickly analyze the validation results and identify
documents that need human review.

Usage:
  python analyze_validation.py <validated.parquet>
"""

import sys
import os

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Install with: pip install pandas")
    sys.exit(1)


def analyze_validation(parquet_file: str):
    """Analyze validation results and generate report."""
    
    if not os.path.exists(parquet_file):
        print(f"Error: File not found: {parquet_file}")
        sys.exit(1)
    
    print(f"{'='*80}")
    print(f"VALIDATION ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"File: {parquet_file}\n")
    
    # Load data
    df = pd.read_parquet(parquet_file)
    
    # Overall statistics
    print(f"📊 OVERALL STATISTICS")
    print(f"{'-'*80}")
    print(f"Total documents: {len(df)}")
    print(f"Total pages: {df['num_pages'].sum()}")
    
    # Check if validation columns exist
    if 'is_aligned' not in df.columns:
        print("\n⚠️  Warning: This file has not been validated yet.")
        print("Run the LLM judge first:")
        print(f"  python llm_judge.py {parquet_file} {parquet_file.replace('.parquet', '_validated.parquet')}")
        return
    
    print(f"\n✅ ALIGNMENT")
    print(f"{'-'*80}")
    aligned = df['is_aligned'].sum()
    print(f"Aligned: {aligned} / {len(df)} ({aligned/len(df)*100:.1f}%)")
    print(f"Misaligned: {len(df) - aligned} / {len(df)} ({(len(df)-aligned)/len(df)*100:.1f}%)")
    
    print(f"\n🎯 REASONABLENESS")
    print(f"{'-'*80}")
    reasonable = df['is_reasonable'].sum()
    print(f"Reasonable: {reasonable} / {len(df)} ({reasonable/len(df)*100:.1f}%)")
    print(f"Unreasonable: {len(df) - reasonable} / {len(df)} ({(len(df)-reasonable)/len(df)*100:.1f}%)")
    
    print(f"\n🔍 CONFIDENCE LEVELS")
    print(f"{'-'*80}")
    avg_confidence = df['confidence'].mean()
    high_conf = (df['confidence'] >= 0.9).sum()
    med_conf = ((df['confidence'] >= 0.7) & (df['confidence'] < 0.9)).sum()
    low_conf = (df['confidence'] < 0.7).sum()
    
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"  High (≥0.9):  {high_conf} docs ({high_conf/len(df)*100:.1f}%)")
    print(f"  Medium (≥0.7): {med_conf} docs ({med_conf/len(df)*100:.1f}%)")
    print(f"  Low (<0.7):   {low_conf} docs ({low_conf/len(df)*100:.1f}%)")
    
    # Documents needing review
    print(f"\n⚠️  DOCUMENTS NEEDING HUMAN REVIEW")
    print(f"{'-'*80}")
    needs_review = df[(df['confidence'] < 0.7) | (~df['is_aligned'])]
    print(f"Total needing review: {len(needs_review)}\n")
    
    if len(needs_review) > 0:
        for idx, row in needs_review.iterrows():
            print(f"{idx+1}. {row['pdf_url']}")
            print(f"   Extracted: {row['money_amounts']}")
            print(f"   LLM found: {row.get('amounts_found_by_llm', 'N/A')}")
            
            if row.get('missing_amounts'):
                print(f"   ⚠️  Missing: {row['missing_amounts']}")
            
            if row.get('incorrect_amounts'):
                print(f"   ❌ Incorrect: {row['incorrect_amounts']}")
            
            print(f"   Confidence: {row['confidence']:.2f}")
            print(f"   Note: {row['note']}")
            print()
    else:
        print("✓ No documents need review!")
    
    # OCR accuracy analysis
    print(f"\n📈 OCR ACCURACY ANALYSIS")
    print(f"{'-'*80}")
    
    # Count documents with missing amounts
    has_missing = df['missing_amounts'].fillna('').str.len() > 0
    print(f"Documents with missing amounts: {has_missing.sum()} / {len(df)}")
    
    # Count documents with incorrect amounts
    has_incorrect = df['incorrect_amounts'].fillna('').str.len() > 0
    print(f"Documents with incorrect amounts: {has_incorrect.sum()} / {len(df)}")
    
    # Perfect extractions (aligned, reasonable, high confidence, no missing/incorrect)
    perfect = df[
        (df['is_aligned']) & 
        (df['is_reasonable']) & 
        (df['confidence'] >= 0.9) &
        (~has_missing) &
        (~has_incorrect)
    ]
    print(f"Perfect extractions: {len(perfect)} / {len(df)} ({len(perfect)/len(df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"✓ {aligned}/{len(df)} aligned")
    print(f"✓ {reasonable}/{len(df)} reasonable")
    print(f"⚠️  {len(needs_review)}/{len(df)} need review")
    print(f"📊 {avg_confidence:.0%} average confidence")
    print(f"✨ {len(perfect)}/{len(df)} perfect extractions")
    print(f"{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_validation.py <validated.parquet>")
        print("\nExample:")
        print("  python analyze_validation.py validated_results.parquet")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    analyze_validation(parquet_file)


if __name__ == "__main__":
    main()

