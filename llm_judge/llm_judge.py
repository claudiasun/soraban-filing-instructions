#!/usr/bin/env python3
"""
LLM Judge for OCR Money Amount Validation

This script uses an LLM to validate money amounts extracted from PDFs using PaddleOCR.
It adds validation columns to the parquet file:
- is_aligned: True/False if values in image align with extracted data
- is_reasonable: True/False if values fall in reasonable range
- llm_note: Explanation if not aligned or unreasonable
- confidence: 0-1 score indicating if human verification is needed
- value_bbox: Bounding boxes of money amounts in the image

Usage:
  python llm_judge.py <input.parquet> [output.parquet] [--provider openai|anthropic|google]
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Install it with: pip install pandas")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: pdf2image is not installed. Install it with: pip install pdf2image")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Warning: PaddleOCR not installed. Bounding box extraction will be limited.")
    PaddleOCR = None


class LLMJudge:
    """LLM-based judge for validating OCR money amount extraction."""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize LLM judge.
        
        Args:
            provider: "openai", "anthropic", or "google"
            model: Specific model name (optional, uses defaults)
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
                self.model = model or "gpt-4o"
            except ImportError:
                print("Error: openai package not installed. Install with: pip install openai")
                sys.exit(1)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                print("Make sure OPENAI_API_KEY environment variable is set")
                sys.exit(1)
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.model = model or "claude-3-5-sonnet-20241022"
            except ImportError:
                print("Error: anthropic package not installed. Install with: pip install anthropic")
                sys.exit(1)
            except Exception as e:
                print(f"Error initializing Anthropic client: {e}")
                print("Make sure ANTHROPIC_API_KEY environment variable is set")
                sys.exit(1)
                
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable is not set")
                genai.configure(api_key=api_key)
                self.model = model or "gemini-3-pro-preview"
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                print("Error: google-generativeai package not installed. Install with: pip install google-generativeai")
                sys.exit(1)
            except Exception as e:
                print(f"Error initializing Google Gemini client: {e}")
                print("Make sure GOOGLE_API_KEY environment variable is set")
                sys.exit(1)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'google'")
        
        print(f"✓ Initialized {self.provider} with model {self.model}")
    
    def image_to_base64(self, image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def call_llm_with_vision(self, image, extracted_amounts: str, full_text: str) -> Dict:
        """
        Call LLM with vision to validate extracted money amounts.
        
        Args:
            image: PIL Image object
            extracted_amounts: Comma-separated string of extracted money amounts
            full_text: Full OCR text extracted from the document
            
        Returns:
            Dictionary with validation results
        """
        # Prepare the prompt
        prompt = f"""You are validating money amounts extracted from a financial document using OCR.

EXTRACTED MONEY AMOUNTS (by OCR):
{extracted_amounts if extracted_amounts else "(None extracted)"}

OCR FULL TEXT:
{full_text[:2000]}{"..." if len(full_text) > 2000 else ""}

Please analyze the image and provide a JSON response with the following fields:

1. "is_aligned": true or false - Do the extracted amounts align with what you see in the image?
2. "is_reasonable": true or false - Are the amounts reasonable for a tax/financial document?
3. "llm_note": string - Explanation if not aligned or unreasonable. Include what amounts you actually see.
4. "confidence": number between 0 and 1 - How confident are you? (1 = very confident, 0 = needs human review)
5. "amounts_found": list of strings - All money amounts you can identify in the image
6. "missing_amounts": list of strings - Amounts you see but weren't extracted
7. "incorrect_amounts": list of strings - Amounts that were extracted but appear wrong

Please be thorough and look for:
- Payment amounts
- Refund amounts
- Tax owed/due
- Penalties and interest
- Overpayments
- Balance amounts
- Automatic withdrawals/deposits

Return ONLY a valid JSON object, no other text."""

        if self.provider == "openai":
            return self._call_openai(image, prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(image, prompt)
        else:
            return self._call_google(image, prompt)
    
    def _call_openai(self, image, prompt: str) -> Dict:
        try:
            image_base64 = self.image_to_base64(image)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]            )
            
            content = response.choices[0].message.content
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "is_aligned": False,
                "is_reasonable": False,
                "llm_note": f"Error calling LLM: {str(e)}",
                "confidence": 0.0,
                "amounts_found": [],
                "missing_amounts": [],
                "incorrect_amounts": []
            }
    
    def _call_anthropic(self, image, prompt: str) -> Dict:
        """Call Anthropic Claude Vision API."""
        try:
            image_base64 = self.image_to_base64(image)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            content = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
                
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return {
                "is_aligned": False,
                "is_reasonable": False,
                "llm_note": f"Error calling LLM: {str(e)}",
                "confidence": 0.0,
                "amounts_found": [],
                "missing_amounts": [],
                "incorrect_amounts": []
            }
    
    def _call_google(self, image, prompt: str) -> Dict:
        """Call Google Gemini Vision API."""
        try:
            import google.generativeai as genai
            
            # Gemini expects PIL Image directly
            response = self.client.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2000,
                )
            )
            
            content = response.text
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
                
        except Exception as e:
            print(f"Error calling Google Gemini API: {e}")
            return {
                "is_aligned": False,
                "is_reasonable": False,
                "llm_note": f"Error calling LLM: {str(e)}",
                "confidence": 0.0,
                "amounts_found": [],
                "missing_amounts": [],
                "incorrect_amounts": []
            }


def concatenate_images_vertically(images: List) -> Image.Image:
    """
    Concatenate multiple PIL images vertically into a single image.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Single PIL Image with all images stacked vertically
    """
    if not images:
        raise ValueError("No images to concatenate")
    
    if len(images) == 1:
        return images[0]
    
    # Calculate dimensions for the concatenated image
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    # Use the maximum width and sum of heights
    max_width = max(widths)
    total_height = sum(heights)
    
    # Create new image with white background
    combined_image = Image.new('RGB', (max_width, total_height), color='white')
    
    # Paste each image
    y_offset = 0
    for img in images:
        # Center the image horizontally if it's narrower than max_width
        x_offset = (max_width - img.width) // 2
        combined_image.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    return combined_image


def extract_bounding_boxes(pdf_path: str) -> Dict[int, List[Dict]]:
    """
    Extract bounding boxes for money amounts from PDF using PaddleOCR.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary mapping page number to list of bounding boxes for money amounts
    """
    if PaddleOCR is None:
        print("Warning: PaddleOCR not available, skipping bbox extraction")
        return {}
    
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        images = convert_from_path(pdf_path, dpi=200)
        
        all_bboxes = {}
        
        for page_num, image in enumerate(images, start=1):
            image_np = np.array(image)
            result = ocr.ocr(image_np, cls=True)
            
            page_bboxes = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # Check if text contains money amount
                        if re.search(r'\$\d+|(?:refund|payment|amount|balance|due|owe|paid|tax)\s*\d+', text, re.IGNORECASE):
                            page_bboxes.append({
                                'bbox': bbox,
                                'text': text,
                                'confidence': confidence,
                                'page': page_num
                            })
            
            if page_bboxes:
                all_bboxes[page_num] = page_bboxes
        
        return all_bboxes
        
    except Exception as e:
        print(f"Warning: Error extracting bboxes: {e}")
        return {}


def validate_parquet(input_path: str, output_path: str, provider: str = "openai", 
                     pdf_directory: Optional[str] = None):
    """
    Validate OCR results in parquet file using LLM judge.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        provider: LLM provider ("openai" or "anthropic")
        pdf_directory: Directory containing PDF files (optional)
    """
    # Load parquet file
    print(f"Loading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"✓ Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize LLM judge
    judge = LLMJudge(provider=provider)
    
    # Initialize new columns
    df['is_aligned'] = None
    df['is_reasonable'] = None
    df['llm_note'] = None
    df['confidence'] = None
    df['value_bbox'] = None
    df['amounts_found_by_llm'] = None
    df['missing_amounts'] = None
    df['incorrect_amounts'] = None
    
    # Process each row
    for idx, row in df.iterrows():
        pdf_filename = row['pdf_url']
        money_amounts = row.get('money_amounts', '')
        full_text = row.get('full_text', '')
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(df)}] Processing: {pdf_filename}")
        print(f"{'='*80}")
        print(f"Extracted amounts: {money_amounts}")
        
        # Find PDF file
        if pdf_directory:
            pdf_path = os.path.join(pdf_directory, pdf_filename)
        else:
            # Try to find in current directory or subdirectories
            search_paths = [
                pdf_filename,
                os.path.join('Filing instructions', pdf_filename),
                os.path.join('..', pdf_filename),
            ]
            pdf_path = None
            for path in search_paths:
                if os.path.exists(path):
                    pdf_path = path
                    break
        
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"⚠ Warning: PDF not found: {pdf_filename}")
            df.at[idx, 'is_aligned'] = False
            df.at[idx, 'is_reasonable'] = False
            df.at[idx, 'llm_note'] = f"PDF file not found: {pdf_filename}"
            df.at[idx, 'confidence'] = 0.0
            df.at[idx, 'value_bbox'] = "[]"
            continue
        
        print(f"Found PDF: {pdf_path}")
        
        # Extract bounding boxes
        print("Extracting bounding boxes...")
        bboxes = extract_bounding_boxes(pdf_path)
        df.at[idx, 'value_bbox'] = json.dumps(bboxes)
        
        # Convert PDF to images for LLM validation
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=200)
        
        # Combine all pages into single validation
        if len(images) == 1:
            main_image = images[0]
            print("Single-page document")
        else:
            # For multi-page, concatenate all pages vertically
            print(f"Multi-page document ({len(images)} pages), concatenating all pages...")
            main_image = concatenate_images_vertically(images)
        
        # Call LLM judge
        print("Calling LLM for validation...")
        validation_result = judge.call_llm_with_vision(main_image, money_amounts, full_text)
        
        # Update dataframe
        df.at[idx, 'is_aligned'] = validation_result.get('is_aligned', False)
        df.at[idx, 'is_reasonable'] = validation_result.get('is_reasonable', False)
        df.at[idx, 'llm_note'] = validation_result.get('llm_note', '')
        df.at[idx, 'confidence'] = validation_result.get('confidence', 0.0)
        df.at[idx, 'amounts_found_by_llm'] = ', '.join(validation_result.get('amounts_found', []))
        df.at[idx, 'missing_amounts'] = ', '.join(validation_result.get('missing_amounts', []))
        df.at[idx, 'incorrect_amounts'] = ', '.join(validation_result.get('incorrect_amounts', []))
        
        print(f"✓ Validation complete:")
        print(f"  - Aligned: {validation_result.get('is_aligned')}")
        print(f"  - Reasonable: {validation_result.get('is_reasonable')}")
        print(f"  - Confidence: {validation_result.get('confidence')}")
        print(f"  - LLM Note: {validation_result.get('llm_note', 'N/A')}")
    
    # Save updated parquet
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"✓ Successfully saved validated results!")
    
    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY:")
    print(f"{'='*80}")
    print(f"Total documents: {len(df)}")
    print(f"Aligned: {df['is_aligned'].sum()}/{len(df)}")
    print(f"Reasonable: {df['is_reasonable'].sum()}/{len(df)}")
    print(f"Avg confidence: {df['confidence'].mean():.2f}")
    print(f"Need human review (confidence < 0.7): {(df['confidence'] < 0.7).sum()}/{len(df)}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Judge for validating OCR money amount extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with OpenAI (default)
  python llm_judge.py results.parquet validated_results.parquet
  
  # Validate with Anthropic Claude
  python llm_judge.py results.parquet validated_results.parquet --provider anthropic
  
  # Validate with Google Gemini
  python llm_judge.py results.parquet validated_results.parquet --provider google
  
  # Specify PDF directory
  python llm_judge.py results.parquet validated_results.parquet --pdf-dir "Filing instructions"
        """
    )
    
    parser.add_argument("input_parquet", help="Input parquet file with OCR results")
    parser.add_argument("output_parquet", nargs='?', help="Output parquet file with validation (default: input_validated.parquet)")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai",
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--model", help="Specific model name (optional)")
    parser.add_argument("--pdf-dir", help="Directory containing PDF files")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output_parquet:
        base = os.path.splitext(args.input_parquet)[0]
        args.output_parquet = f"{base}_validated.parquet"
    
    # Validate
    validate_parquet(
        args.input_parquet,
        args.output_parquet,
        provider=args.provider,
        pdf_directory=args.pdf_dir
    )


if __name__ == "__main__":
    main()

