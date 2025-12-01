#!/usr/bin/env python3
"""
Test script for LLM judge validation.

This script demonstrates the complete workflow:
1. Check if required packages are installed
2. Validate environment setup
3. Run a small test validation
4. Show results

Usage:
  python test_llm_judge.py [--provider openai|anthropic]
"""

import sys
import os
import argparse


def check_imports():
    """Check if required packages are installed."""
    print("Checking required packages...")
    
    required = {
        'pandas': 'pip install pandas',
        'pyarrow': 'pip install pyarrow',
        'PIL': 'pip install pillow',
        'pdf2image': 'pip install pdf2image'
    }
    
    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - Install with: {install_cmd}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install all with: pip install -r requirements_llm_judge.txt")
        return False
    
    print("✓ All required packages installed\n")
    return True


def check_llm_provider(provider):
    """Check if LLM provider is properly configured."""
    print(f"Checking {provider} configuration...")
    
    if provider == 'openai':
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("  ✗ OPENAI_API_KEY not set")
                print("  Set it with: export OPENAI_API_KEY='sk-...'")
                return False
            
            # Verify the key format
            if not api_key.startswith('sk-'):
                print("  ✗ OPENAI_API_KEY appears invalid (should start with 'sk-')")
                return False
            
            print(f"  ✓ openai package installed")
            print(f"  ✓ OPENAI_API_KEY set ({api_key[:8]}...)")
            return True
            
        except ImportError:
            print("  ✗ openai package not installed")
            print("  Install with: pip install openai")
            return False
    
    elif provider == 'anthropic':
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("  ✗ ANTHROPIC_API_KEY not set")
                print("  Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
                return False
            
            # Verify the key format
            if not api_key.startswith('sk-ant-'):
                print("  ✗ ANTHROPIC_API_KEY appears invalid (should start with 'sk-ant-')")
                return False
            
            print(f"  ✓ anthropic package installed")
            print(f"  ✓ ANTHROPIC_API_KEY set ({api_key[:12]}...)")
            return True
            
        except ImportError:
            print("  ✗ anthropic package not installed")
            print("  Install with: pip install anthropic")
            return False
    
    return False


def find_test_file():
    """Find a test parquet file to validate."""
    print("Looking for test files...")
    
    # Look for any parquet file in current directory or parent
    candidates = [
        "CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet",
        "../CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet",
        "Filing instructions/Filing instructions_ocr_results.parquet",
        "../Filing instructions/Filing instructions_ocr_results.parquet"
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"  ✓ Found: {candidate}")
            return candidate
    
    print("  ✗ No test files found")
    print("\nPlease run OCR first:")
    print("  python run.py 'Filing instructions' test_results.parquet")
    return None


def run_test_validation(provider, test_file):
    """Run a test validation."""
    print(f"\n{'='*80}")
    print(f"RUNNING TEST VALIDATION")
    print(f"{'='*80}\n")
    
    print(f"Provider: {provider}")
    print(f"Input: {test_file}")
    
    try:
        from llm_judge import LLMJudge
        from pdf2image import convert_from_path
        import pandas as pd
        
        # Load the parquet file
        df = pd.read_parquet(test_file)
        print(f"Loaded {len(df)} document(s)\n")
        
        # Get first document
        row = df.iloc[0]
        pdf_filename = row['pdf_url']
        money_amounts = row.get('money_amounts', '')
        full_text = row.get('full_text', '')
        
        print(f"Test document: {pdf_filename}")
        print(f"Extracted amounts: {money_amounts}")
        
        # Find PDF file
        pdf_paths = [
            pdf_filename,
            f"Filing instructions/{pdf_filename}",
            f"../{pdf_filename}",
            f"../Filing instructions/{pdf_filename}"
        ]
        
        pdf_path = None
        for path in pdf_paths:
            if os.path.exists(path):
                pdf_path = path
                break
        
        if not pdf_path:
            print(f"✗ PDF file not found: {pdf_filename}")
            print("  Skipping validation test")
            return False
        
        print(f"Found PDF: {pdf_path}")
        
        # Convert first page to image
        print("\nConverting PDF to image...")
        images = convert_from_path(pdf_path, dpi=200)
        print(f"✓ Converted {len(images)} page(s)")
        
        # Initialize LLM judge
        print(f"\nInitializing {provider} LLM judge...")
        judge = LLMJudge(provider=provider)
        
        # Run validation on first page
        print("Calling LLM for validation (this may take 10-30 seconds)...")
        result = judge.call_llm_with_vision(images[0], money_amounts, full_text[:2000])
        
        # Display results
        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}\n")
        
        print(f"✓ Is Aligned:     {result.get('is_aligned')}")
        print(f"✓ Is Reasonable:  {result.get('is_reasonable')}")
        print(f"✓ Confidence:     {result.get('confidence'):.2f}")
        
        print(f"\n📝 Note:")
        print(f"{result.get('note', 'N/A')}")
        
        if result.get('amounts_found'):
            print(f"\n💰 Amounts found by LLM:")
            for amount in result['amounts_found']:
                print(f"  - {amount}")
        
        if result.get('missing_amounts'):
            print(f"\n⚠️  Missing amounts:")
            for amount in result['missing_amounts']:
                print(f"  - {amount}")
        
        if result.get('incorrect_amounts'):
            print(f"\n❌ Incorrect amounts:")
            for amount in result['incorrect_amounts']:
                print(f"  - {amount}")
        
        print(f"\n{'='*80}")
        print("✓ Test validation completed successfully!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LLM judge setup and run a validation test")
    parser.add_argument('--provider', choices=['openai', 'anthropic'], default='anthropic',
                       help='LLM provider to test (default: anthropic)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("LLM JUDGE TEST SCRIPT")
    print(f"{'='*80}\n")
    
    # Step 1: Check required packages
    print("STEP 1: Checking dependencies")
    print("-" * 80)
    if not check_imports():
        return 1
    
    # Step 2: Check LLM provider
    print("\nSTEP 2: Checking LLM provider")
    print("-" * 80)
    if not check_llm_provider(args.provider):
        return 1
    print()
    
    # Step 3: Find test file
    print("STEP 3: Finding test file")
    print("-" * 80)
    test_file = find_test_file()
    if not test_file:
        return 1
    print()
    
    # Step 4: Run test validation
    print("STEP 4: Running test validation")
    print("-" * 80)
    if not run_test_validation(args.provider, test_file):
        return 1
    
    # Success!
    print("🎉 ALL TESTS PASSED!")
    print("\nYou're ready to use the LLM judge. Try:")
    print(f"  python llm_judge.py {test_file} validated_output.parquet --provider {args.provider}")
    print("\nOr run the full example:")
    print("  python example_usage.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

