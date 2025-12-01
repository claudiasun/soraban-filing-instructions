#!/usr/bin/env python3
"""
Simple script to run PaddleOCR on PDF files.
Usage: 
  python run.py <pdf_file> [output.parquet]           # Process single PDF
  python run.py <directory> [output.parquet]          # Process all PDFs in directory
"""

import sys
import os
import re
import glob
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

def process_pdf(pdf_path, ocr):
    """Process a single PDF file and return its results as a dictionary."""
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(pdf_path)}")
    print('='*80)
    
    pdf_filename = os.path.basename(pdf_path)
    
    try:
        print(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=200)
        print(f"Processing {len(images)} pages...")
        
        # Collect results for all pages
        all_texts = []
        all_money_amounts = []
        
        for page_num, image in enumerate(images, start=1):
            print(f"  Page {page_num}/{len(images)}...", end=' ')
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            result = ocr.predict(image_np)
            
            if result and len(result) > 0:
                ocr_result = result[0]
                # The OCRResult object contains 'rec_texts' key with the recognized text
                if 'rec_texts' in ocr_result and ocr_result['rec_texts']:
                    text_lines = ocr_result['rec_texts']
                    text = "\n".join(text_lines)
                    print(f"✓ ({len(text_lines)} lines)")
                    
                    # Add page separator and text
                    all_texts.append(f"[Page {page_num}]\n{text}")
                    
                    # Extract only money amounts (with $ sign or after common money keywords)
                    # Pattern 1: Explicit dollar signs
                    money_with_dollar = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
                    
                    # Pattern 2: Numbers after money-related keywords
                    money_context = re.findall(
                        r'(?:refund|payment|amount|balance|due|owe|paid|tax|deposit|withdrawal|penalty|interest|overpayment)[\s:]+\$?\d+(?:,\d{3})*(?:\.\d{2})?',
                        text,
                        re.IGNORECASE
                    )
                    # Extract just the number part from context matches
                    money_from_context = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', ' '.join(money_context))
                    
                    # Combine and deduplicate
                    page_money = list(set(money_with_dollar + money_from_context))
                    all_money_amounts.extend(page_money)
                else:
                    print("(No text detected)")
                    all_texts.append(f"[Page {page_num}]\n(No text detected)")
            else:
                print("(No text detected)")
                all_texts.append(f"[Page {page_num}]\n(No text detected)")
        
        # Combine all pages into single row
        combined_text = "\n\n".join(all_texts)
        # Return result data for this PDF
        unique_money = sorted(set(all_money_amounts))
        print(f"✓ Completed - Found {len(unique_money)} unique money amounts")
        
        return {
            'pdf_url': pdf_filename,
            'num_pages': len(images),
            'money_amounts': ', '.join(unique_money),  # Deduplicated and sorted
            'full_text': combined_text
        }
    except Exception as e:
        print(f"✗ Error processing {pdf_filename}: {str(e)}")
        return {
            'pdf_url': pdf_filename,
            'num_pages': 0,
            'money_amounts': '',
            'full_text': f'Error: {str(e)}'
        }


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py <pdf_file> [output.parquet]")
        print("  python run.py <directory> [output.parquet]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Get list of PDF files to process
    pdf_files = []
    if os.path.isfile(input_path):
        # Single file
        pdf_files = [input_path]
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}_ocr_results.parquet"
    elif os.path.isdir(input_path):
        # Directory - find all PDFs
        pdf_files = glob.glob(os.path.join(input_path, "*.pdf"))
        pdf_files.sort()  # Sort for consistent ordering
        if not pdf_files:
            print(f"Error: No PDF files found in {input_path}")
            sys.exit(1)
        if output_path is None:
            dir_name = os.path.basename(input_path.rstrip('/'))
            output_path = f"{dir_name}_ocr_results.parquet"
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize PaddleOCR once
    print("\nInitializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    # Process all PDFs
    all_results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] ", end='')
        result = process_pdf(pdf_path, ocr)
        all_results.append(result)
    
    # Save all results to single parquet file
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    df = pd.DataFrame(all_results)
    df.to_parquet(output_path, index=False)
    print(f"✓ Successfully saved {len(df)} PDF(s) to {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total PDFs processed: {len(df)}")
    print(f"  Total pages: {df['num_pages'].sum()}")
    print(f"\nDone!")

if __name__ == "__main__":
    main()
