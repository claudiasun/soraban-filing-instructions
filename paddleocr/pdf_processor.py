#!/usr/bin/env python3
"""
PaddleOCR PDF Processor

This script demonstrates how to use PaddleOCR to extract text from PDF files.
It converts PDF pages to images and then runs OCR on each image.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCR is not installed. Install it with: pip install paddlepaddle paddleocr")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
except ImportError:
    print("Error: pdf2image is not installed. Install it with: pip install pdf2image")
    print("Note: You also need poppler-utils installed on your system.")
    print("  macOS: brew install poppler")
    print("  Ubuntu: sudo apt-get install poppler-utils")
    sys.exit(1)


class PaddleOCRPDFProcessor:
    """Process PDF files using PaddleOCR for text extraction."""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        """
        Initialize PaddleOCR processor.
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language code ('en', 'ch', etc.)
        """
        print(f"Initializing PaddleOCR with language: {lang}...")
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=False)
        print("PaddleOCR initialized successfully!")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion (default: 200)
        
        Returns:
            List of PIL Image objects, one per page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Converting PDF to images: {pdf_path}...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} pages to images")
        return images
    
    def process_image(self, image, page_num: int) -> Dict:
        """
        Run OCR on a single image.
        
        Args:
            image: PIL Image object
            page_num: Page number (1-indexed)
        
        Returns:
            Dictionary with page number and extracted text
        """
        print(f"Processing page {page_num}...")
        result = self.ocr.ocr(image, cls=True)
        
        # Extract text from OCR result
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text_lines.append(line[1][0])  # Extract text from (bbox, (text, confidence))
        
        full_text = "\n".join(text_lines)
        
        return {
            "page": page_num,
            "text": full_text,
            "line_count": len(text_lines)
        }
    
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None, 
                   save_images: bool = False, dpi: int = 200) -> List[Dict]:
        """
        Process entire PDF file and extract text from all pages.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images (if save_images=True)
            save_images: Whether to save intermediate images to disk
            dpi: Resolution for image conversion
        
        Returns:
            List of dictionaries, each containing page number and extracted text
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi=dpi)
        
        # Create output directory if needed
        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process each page
        results = []
        for idx, image in enumerate(images, start=1):
            # Save image if requested
            if save_images and output_dir:
                image_path = os.path.join(output_dir, f"page_{idx}.png")
                image.save(image_path)
                print(f"Saved image: {image_path}")
            
            # Run OCR on image
            result = self.process_image(image, idx)
            results.append(result)
            
            print(f"Page {idx}: Extracted {result['line_count']} lines of text")
        
        return results
    
    def process_pdf_to_text(self, pdf_path: str, dpi: int = 200) -> str:
        """
        Process PDF and return all text concatenated.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion
        
        Returns:
            Complete text from all pages
        """
        results = self.process_pdf(pdf_path, dpi=dpi)
        all_text = "\n\n".join([f"=== Page {r['page']} ===\n{r['text']}" for r in results])
        return all_text


def main():
    """Main function to run the script from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from PDF using PaddleOCR")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output file for extracted text (optional)")
    parser.add_argument("--save-images", action="store_true", help="Save intermediate images")
    parser.add_argument("--image-dir", default="pdf_images", help="Directory to save images (default: pdf_images)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for image conversion (default: 200)")
    parser.add_argument("--lang", default="en", help="Language code for OCR (default: en)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PaddleOCRPDFProcessor(lang=args.lang)
    
    # Process PDF
    print(f"\n{'='*60}")
    print(f"Processing PDF: {args.pdf_path}")
    print(f"{'='*60}\n")
    
    results = processor.process_pdf(
        args.pdf_path,
        output_dir=args.image_dir if args.save_images else None,
        save_images=args.save_images,
        dpi=args.dpi
    )
    
    # Output results
    if args.json:
        import json
        output = json.dumps(results, indent=2)
    else:
        output = "\n\n".join([
            f"{'='*60}\nPage {r['page']} ({r['line_count']} lines)\n{'='*60}\n{r['text']}"
            for r in results
        ])
    
    # Write to file or print to stdout
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n{'='*60}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("EXTRACTED TEXT:")
        print(f"{'='*60}\n")
        print(output)
    
    # Print summary
    total_pages = len(results)
    total_lines = sum(r['line_count'] for r in results)
    total_chars = sum(len(r['text']) for r in results)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total pages processed: {total_pages}")
    print(f"  Total lines extracted: {total_lines}")
    print(f"  Total characters: {total_chars:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
