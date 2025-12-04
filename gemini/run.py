#!/usr/bin/env python3
"""
Gemini 2.5 Pro Tax Payment Extractor with LLM Judge Validation

Uses Google's Gemini 2.5 Pro vision model to extract payment-related information
from tax filing instruction PDFs, with optional LLM-based validation.

Usage: 
  python run.py <pdf_file> [output.parquet]                    # Extract only
  python run.py <pdf_file> [output.parquet] --validate         # Extract + validate
  python run.py <directory> [output.parquet] --validate        # Batch with validation
  python run.py <input.parquet> --validate-only                # Validate existing results

Environment:
  GOOGLE_API_KEY    - For Gemini extraction and validation
  OPENAI_API_KEY    - For OpenAI validation (if --judge-provider openai)
  ANTHROPIC_API_KEY - For Anthropic validation (if --judge-provider anthropic)
"""

import sys
import os
import re
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO
import base64

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Install with: pip install pandas")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Error: pdf2image/Pillow not installed. Install with: pip install pdf2image Pillow")
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Install with: pip install google-generativeai")
    sys.exit(1)


def generate_prompt_for_extracting_tax_payment_info() -> str:
    """Generate the prompt for extracting tax payment information."""
    return """
  Extract ONLY payment-related information from this tax filing instruction PDF or text.  
  Ignore all details not related to payment, refund, or overpayment amounts.

  Extract the following fields:

  - federal_amount_due: Numeric amount the taxpayer must PAY for federal taxes  
                        (no commas or currency symbols; null if no payment is required)

  - federal_overpayment: Numeric amount the taxpayer is receiving as a REFUND or overpayment  
                         (null if not applicable)

  - federal_payment_due_date: The due date for federal tax payments in yyyy-MM-dd format  
                              (null if not found or unclear)

  - state_payments: A list of payment objects. For each state, extract:
        - state: State name or abbreviation
        - amount_due: Numeric amount owed (null if refund or not applicable)
        - overpayment: Numeric refund/overpayment amount (null if not applicable)
        - payment_due_date: yyyy-MM-dd format or null
        - payment_link: URL for making the payment (or null)

  - payment_methods: Any described payment methods (e.g., online portal, mail-in check, EFT).  
                     If not mentioned, return null.

  Return ONLY a valid JSON object with this structure:

  {
    \"federal_amount_due\": numeric_or_null,
    \"federal_overpayment\": numeric_or_null,
    \"federal_payment_due_date\": \"yyyy-MM-dd or null\",
    \"state_payments\": [
      {
        \"state\": \"CA or NY or etc\",
        \"amount_due\": numeric_or_null,
        \"overpayment\": numeric_or_null,
        \"payment_due_date\": \"yyyy-MM-dd or null\",
        \"payment_link\": \"value or null\"
      }
    ],
    \"payment_methods\": \"value or null\"
  }

  IMPORTANT:
  - Convert values like \"$1,234.56\" or \"1,234\" to plain numbers (e.g., 1234.56 or 1234).
  - If a refund or overpayment is mentioned, treat it as overpayment, NOT a payment due.
  - If both payment and refund appear, extract both separately.
  - If date formats vary (e.g., \"April 18, 2025\" or \"4/18/25\"), convert to yyyy-MM-dd.
  - If a field cannot be found, set it to null.
  - DO NOT include any text outside the JSON.
  "
"""


class GeminiOCR:
    """Gemini 2.5 Pro based extractor for tax payment information."""
    
    def __init__(self, model: str = "gemini-2.5-pro"):
        """
        Initialize Gemini extractor.
        
        Args:
            model: Gemini model name (default: gemini-2.5-pro)
        """
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable is not set")
            print("Get your API key from: https://aistudio.google.com/apikey")
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        print(f"✓ Initialized Gemini with model: {model}")
    
    def resize_image_if_needed(self, image: Image.Image, max_pixels: int = 16_000_000) -> Image.Image:
        """
        Resize image if it exceeds maximum pixel count (Gemini limit).
        
        Args:
            image: PIL Image object
            max_pixels: Maximum number of pixels (default: 16M for Gemini)
            
        Returns:
            Resized PIL Image if needed, otherwise original
        """
        current_pixels = image.width * image.height
        
        if current_pixels <= max_pixels:
            return image
        
        scale = (max_pixels / current_pixels) ** 0.5
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def extract_from_image(self, image: Image.Image) -> Dict:
        """
        Extract tax payment information from a single image using Gemini vision.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with extracted payment information
        """
        prompt = generate_prompt_for_extracting_tax_payment_info()

        try:
            # Resize if needed
            image = self.resize_image_if_needed(image)
            
            # Safety settings for financial documents
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
            
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
                safety_settings=safety_settings
            )
            
            # Check for blocked content
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return {
                    "federal_amount_due": None,
                    "federal_payment_due_date": None,
                    "state_payments": [],
                    "payment_methods": None,
                    "error": str(response.prompt_feedback.block_reason)
                }
            
            if not response.candidates:
                return {
                    "federal_amount_due": None,
                    "federal_payment_due_date": None,
                    "state_payments": [],
                    "payment_methods": None,
                    "error": "No candidates returned"
                }
            
            # Parse JSON response
            content = response.text
            
            # Clean up response - remove markdown code blocks if present
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)
            content = content.strip()
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "federal_amount_due": None,
                    "federal_payment_due_date": None,
                    "state_payments": [],
                    "payment_methods": None,
                    "error": "No valid JSON in response",
                    "raw_response": content[:500]
                }
                
        except json.JSONDecodeError as e:
            return {
                "federal_amount_due": None,
                "federal_payment_due_date": None,
                "state_payments": [],
                "payment_methods": None,
                "error": f"JSON parse error: {str(e)}",
                "raw_response": content[:500] if 'content' in locals() else None
            }
        except Exception as e:
            return {
                "federal_amount_due": None,
                "federal_payment_due_date": None,
                "state_payments": [],
                "payment_methods": None,
                "error": str(e)
            }
    
    def extract_from_images(self, images: List[Image.Image]) -> Dict:
        """
        Extract tax payment info from multiple page images (concatenated into one).
        
        Args:
            images: List of PIL Image objects (one per page)
            
        Returns:
            Dictionary with extracted payment information
        """
        if len(images) == 1:
            return self.extract_from_image(images[0])
        
        # Concatenate all pages vertically for multi-page documents
        combined = self._concatenate_images(images)
        return self.extract_from_image(combined)
    
    def _concatenate_images(self, images: List[Image.Image]) -> Image.Image:
        """Concatenate multiple images vertically."""
        if not images:
            raise ValueError("No images to concatenate")
        
        if len(images) == 1:
            return images[0]
        
        # Calculate dimensions
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        # Create new image with white background
        combined = Image.new('RGB', (max_width, total_height), color='white')
        
        # Paste each image
        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height
        
        return combined


class LLMJudge:
    """LLM-based judge for validating tax payment extraction results."""
    
    def __init__(self, provider: str = "google", model: Optional[str] = None):
        """
        Initialize LLM judge.
        
        Args:
            provider: "google", "openai", or "anthropic"
            model: Specific model name (optional, uses defaults)
        """
        self.provider = provider.lower()
        
        if self.provider == "google":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            genai.configure(api_key=api_key)
            self.model_name = model or "gemini-2.5-pro"
            self.client = genai.GenerativeModel(self.model_name)
            
        elif self.provider == "openai":
            try:
                import openai
                self.openai = openai
                self.client = openai.OpenAI()
                self.model_name = model or "gpt-4o"
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.anthropic = anthropic
                self.client = anthropic.Anthropic()
                self.model_name = model or "claude-sonnet-4-20250514"
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'google', 'openai', or 'anthropic'")
        
        print(f"✓ Initialized LLM Judge with {self.provider} ({self.model_name})")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _resize_image(self, image: Image.Image, max_pixels: int = 16_000_000) -> Image.Image:
        """Resize image if needed."""
        current_pixels = image.width * image.height
        if current_pixels <= max_pixels:
            return image
        scale = (max_pixels / current_pixels) ** 0.5
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def generate_validation_prompt(self, extracted_data: Dict) -> str:
        """Generate the validation prompt."""
        return f"""You are validating tax payment information extracted from a filing instruction document.

    EXTRACTED DATA TO VALIDATE:
    {json.dumps(extracted_data, indent=2)}

    Please analyze the document image and validate the extracted information. Check:

    1. **Federal Payment**: Is the federal_amount_due correct? Is the federal_payment_due_date accurate?
    2. **State Payments**: Are all state payments correctly identified with accurate amounts and dates?
    3. **Missing Information**: Are there any payments visible in the document that were NOT extracted?
    4. **Incorrect Values**: Are any extracted values wrong (wrong amount, wrong date, wrong state)?
    5. **Refund vs Payment**: If the document shows a refund, was it correctly set to null?

    Return ONLY a valid JSON object with this structure:
    {{
    "is_accurate": true or false,
    "confidence": 0.0 to 1.0,
    "federal_validation": {{
        "amount_correct": true/false/null,
        "date_correct": true/false/null,
        "actual_amount": number or null,
        "actual_date": "yyyy-MM-dd" or null,
        "note": "explanation if incorrect"
    }},
    "state_validation": [
        {{
        "state": "CA",
        "amount_correct": true/false,
        "date_correct": true/false,
        "actual_amount": number or null,
        "actual_date": "yyyy-MM-dd" or null,
        "note": "explanation if incorrect"
        }}
    ],
    "missing_payments": [
        {{
        "type": "federal" or "state",
        "state": "XX" (if state),
        "amount": number,
        "date": "yyyy-MM-dd" or null
        }}
    ],
    "llm_judge_note": "Overall assessment and any issues found"
    }}

    IMPORTANT:
    - Be thorough - check every visible payment amount in the document
    - confidence should be 1.0 if very certain, lower if document is unclear
    - Set is_accurate to false if ANY payment is missing or incorrect
    - DO NOT include any text outside the JSON object
    """
    
    def validate(self, image: Image.Image, extracted_data: Dict) -> Dict:
        """
        Validate extracted data against the document image.
        
        Args:
            image: PIL Image of the document
            extracted_data: Dictionary with extracted payment information
            
        Returns:
            Dictionary with validation results
        """
        prompt = self.generate_validation_prompt(extracted_data)
        image = self._resize_image(image)
        
        try:
            if self.provider == "google":
                return self._validate_google(image, prompt)
            elif self.provider == "openai":
                return self._validate_openai(image, prompt)
            else:
                return self._validate_anthropic(image, prompt)
        except Exception as e:
            return {
                "is_accurate": False,
                "confidence": 0.0,
                "llm_judge_note": f"Validation error: {str(e)}",
                "error": str(e)
            }
    
    def _validate_google(self, image: Image.Image, prompt: str) -> Dict:
        """Validate using Google Gemini."""
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        response = self.client.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
            safety_settings=safety_settings
        )
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return {
                "is_accurate": False,
                "confidence": 0.0,
                "llm_judge_note": f"Blocked by safety filters: {response.prompt_feedback.block_reason}"
            }
        
        content = response.text
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"is_accurate": False, "confidence": 0.0, "llm_judge_note": "No valid JSON response"}
    
    def _validate_openai(self, image: Image.Image, prompt: str) -> Dict:
        """Validate using OpenAI."""
        image_base64 = self._image_to_base64(image)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"is_accurate": False, "confidence": 0.0, "llm_judge_note": "No valid JSON response"}
    
    def _validate_anthropic(self, image: Image.Image, prompt: str) -> Dict:
        """Validate using Anthropic Claude."""
        image_base64 = self._image_to_base64(image)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4000,
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
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        content = response.content[0].text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"is_accurate": False, "confidence": 0.0, "llm_judge_note": "No valid JSON response"}


def process_pdf(pdf_path: str, ocr: GeminiOCR, dpi: int = 200, 
                 judge: Optional[LLMJudge] = None, base_dir: Optional[str] = None) -> Dict:
    """
    Process a single PDF file to extract tax payment information.
    
    Args:
        pdf_path: Path to PDF file
        ocr: GeminiOCR instance
        dpi: DPI for PDF to image conversion
        judge: Optional LLMJudge for validation
        base_dir: Base directory for relative path calculation
        
    Returns:
        Dictionary with extracted payment information (and validation if judge provided)
    """
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(pdf_path)}")
    print('='*80)

    # Calculate relative path from base_dir (e.g., "./filename.pdf")
    if base_dir:
        pdf_relative_path = "./" + os.path.relpath(pdf_path, base_dir)
    else:
        pdf_relative_path = "./" + os.path.basename(pdf_path)
    try:
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Extracting payment info from {len(images)} page(s)...")
        
        # Extract from all pages at once
        result = ocr.extract_from_images(images)
        
        # Check for errors
        if result.get('error'):
            print(f"⚠ Warning: {result['error']}")
        
        # Format output
        federal_amount = result.get('federal_amount_due')
        federal_date = result.get('federal_payment_due_date')
        state_payments = result.get('state_payments', [])
        payment_methods = result.get('payment_methods')
        
        print(f"✓ Extraction complete:")
        if federal_amount is not None:
            print(f"  Federal amount due: ${federal_amount:,.2f}" if isinstance(federal_amount, (int, float)) else f"  Federal amount due: {federal_amount}")
        else:
            print(f"  Federal amount due: None (refund or not found)")
        
        if federal_date:
            print(f"  Federal due date: {federal_date}")
        
        if state_payments:
            print(f"  State payments: {len(state_payments)} state(s)")
            for sp in state_payments:
                state = sp.get('state', 'Unknown')
                amt = sp.get('amount_due')
                if amt is not None:
                    print(f"    - {state}: ${amt:,.2f}" if isinstance(amt, (int, float)) else f"    - {state}: {amt}")
                else:
                    print(f"    - {state}: None (refund or not found)")
        
        output = {
            'pdf_url': pdf_relative_path,
            'num_pages': len(images),
            'federal_amount_due': federal_amount,
            'federal_payment_due_date': federal_date,
            'state_payments': json.dumps(state_payments) if state_payments else '[]',
            'payment_methods': payment_methods,
            'error': result.get('error'),
            'raw_response': result.get('raw_response')
        }
        
        # Validate if judge is provided
        if judge:
            print(f"  Validating with LLM Judge...")
            # Concatenate images for validation
            if len(images) == 1:
                combined_image = images[0]
            else:
                combined_image = ocr._concatenate_images(images)
            
            # Prepare extracted data for validation
            extracted_data = {
                'federal_amount_due': federal_amount,
                'federal_payment_due_date': federal_date,
                'state_payments': state_payments,
                'payment_methods': payment_methods
            }
            
            validation = judge.validate(combined_image, extracted_data)
            
            # Add validation results to output
            output['is_accurate'] = validation.get('is_accurate', False)
            output['confidence'] = validation.get('confidence', 0.0)
            output['llm_judge_note'] = validation.get('llm_judge_note', '')
            output['federal_validation'] = json.dumps(validation.get('federal_validation', {}))
            output['state_validation'] = json.dumps(validation.get('state_validation', []))
            output['missing_payments'] = json.dumps(validation.get('missing_payments', []))
            
            # Print validation summary
            is_accurate = validation.get('is_accurate', False)
            confidence = validation.get('confidence', 0.0)
            status = "✓" if is_accurate else "⚠"
            print(f"  {status} Validation: {'ACCURATE' if is_accurate else 'ISSUES FOUND'} (confidence: {confidence:.1%})")
            if validation.get('llm_judge_note'):
                print(f"    Note: {validation.get('llm_judge_note')[:100]}...")
        
        return output
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        output = {
            'pdf_url': pdf_relative_path,
            'num_pages': 0,
            'federal_amount_due': None,
            'federal_payment_due_date': None,
            'state_payments': '[]',
            'payment_methods': None,
            'error': str(e),
            'raw_response': None
        }
        if judge:
            output.update({
                'is_accurate': False,
                'confidence': 0.0,
                'llm_judge_note': f"Processing error: {str(e)}",
                'federal_validation': '{}',
                'state_validation': '[]',
                'missing_payments': '[]'
            })
        return output


def validate_existing_parquet(input_path: str, output_path: str, 
                               pdf_directory: str, judge: LLMJudge, dpi: int = 200):
    """
    Validate an existing parquet file with extracted payment information.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        pdf_directory: Directory containing the PDF files
        judge: LLMJudge instance
        dpi: DPI for PDF conversion
    """
    print(f"Loading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"✓ Loaded {len(df)} rows")
    
    # Initialize validation columns
    df['is_accurate'] = None
    df['confidence'] = None
    df['llm_judge_note'] = None
    df['federal_validation'] = None
    df['state_validation'] = None
    df['missing_payments'] = None
    
    for idx, row in df.iterrows():
        pdf_url = row['pdf_url']
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(df)}] Validating: {pdf_url}")
        print('='*80)
        
        # Find PDF file - try multiple paths
        pdf_path = None
        candidates = [
            pdf_url,  # Try relative path directly
            os.path.join(pdf_directory, pdf_url),  # Join with pdf_dir
            os.path.join(pdf_directory, os.path.basename(pdf_url)),  # Just filename in pdf_dir
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                pdf_path = candidate
                break
        
        if not pdf_path:
            print(f"⚠ PDF not found: {pdf_url}")
            df.at[idx, 'is_accurate'] = False
            df.at[idx, 'confidence'] = 0.0
            df.at[idx, 'llm_judge_note'] = f"PDF not found: {pdf_url}"
            continue
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # Concatenate images
            if len(images) == 1:
                combined_image = images[0]
            else:
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                combined_image = Image.new('RGB', (max_width, total_height), color='white')
                y_offset = 0
                for img in images:
                    x_offset = (max_width - img.width) // 2
                    combined_image.paste(img, (x_offset, y_offset))
                    y_offset += img.height
            
            # Prepare extracted data
            state_payments = row.get('state_payments', '[]')
            if isinstance(state_payments, str):
                state_payments = json.loads(state_payments)
            
            extracted_data = {
                'federal_amount_due': row.get('federal_amount_due'),
                'federal_payment_due_date': row.get('federal_payment_due_date'),
                'state_payments': state_payments,
                'payment_methods': row.get('payment_methods')
            }
            
            # Validate
            validation = judge.validate(combined_image, extracted_data)
            
            # Update dataframe
            df.at[idx, 'is_accurate'] = validation.get('is_accurate', False)
            df.at[idx, 'confidence'] = validation.get('confidence', 0.0)
            df.at[idx, 'llm_judge_note'] = validation.get('llm_judge_note', '')
            df.at[idx, 'federal_validation'] = json.dumps(validation.get('federal_validation', {}))
            df.at[idx, 'state_validation'] = json.dumps(validation.get('state_validation', []))
            df.at[idx, 'missing_payments'] = json.dumps(validation.get('missing_payments', []))
            
            is_accurate = validation.get('is_accurate', False)
            confidence = validation.get('confidence', 0.0)
            status = "✓" if is_accurate else "⚠"
            print(f"{status} {'ACCURATE' if is_accurate else 'ISSUES FOUND'} (confidence: {confidence:.1%})")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            df.at[idx, 'is_accurate'] = False
            df.at[idx, 'confidence'] = 0.0
            df.at[idx, 'llm_judge_note'] = f"Error: {str(e)}"
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving validated results to: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"✓ Saved {len(df)} rows")
    
    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    accurate_count = df['is_accurate'].sum()
    print(f"  Total documents: {len(df)}")
    print(f"  Accurate: {accurate_count}/{len(df)} ({accurate_count/len(df)*100:.1f}%)")
    print(f"  Issues found: {len(df) - accurate_count}/{len(df)}")
    print(f"  Average confidence: {df['confidence'].mean():.1%}")
    print(f"  Low confidence (<70%): {(df['confidence'] < 0.7).sum()}/{len(df)}")
    print(f"{'='*80}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gemini 2.5 Pro Tax Payment Extractor with LLM Judge Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract payment info from PDFs
  python run.py "Filing instructions/" results.parquet
  
  # Extract + validate in one pass
  python run.py "Filing instructions/" results.parquet --validate
  
  # Use different judge provider (cross-validate with Claude)
  python run.py document.pdf output.parquet --validate --judge-provider anthropic
  
  # Validate existing parquet file
  python run.py existing_results.parquet --validate-only --pdf-dir "Filing instructions/"

Environment:
  GOOGLE_API_KEY    - For Gemini extraction and validation
  OPENAI_API_KEY    - For OpenAI validation (if --judge-provider openai)
  ANTHROPIC_API_KEY - For Anthropic validation (if --judge-provider anthropic)
        """
    )
    
    parser.add_argument("input_path", help="PDF file, directory, or parquet file (with --validate-only)")
    parser.add_argument("output_parquet", nargs='?', help="Output parquet file (default: auto-generated)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion (default: 200)")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model for extraction (default: gemini-2.5-pro)")
    
    # Validation options
    parser.add_argument("--validate", action="store_true", help="Enable LLM judge validation after extraction")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing parquet (skip extraction)")
    parser.add_argument("--judge-provider", choices=["google", "openai", "anthropic"], default="google",
                        help="LLM provider for validation (default: google)")
    parser.add_argument("--judge-model", help="Specific model for judge (optional)")
    parser.add_argument("--pdf-dir", help="PDF directory (required for --validate-only)")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_parquet
    
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Handle --validate-only mode
    if args.validate_only:
        if not input_path.endswith('.parquet'):
            print("Error: --validate-only requires a parquet file as input")
            sys.exit(1)
        if not args.pdf_dir:
            print("Error: --validate-only requires --pdf-dir to locate PDFs")
            sys.exit(1)
        if not os.path.exists(args.pdf_dir):
            print(f"Error: PDF directory not found: {args.pdf_dir}")
            sys.exit(1)
        
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_validated.parquet"
        
        print(f"{'='*80}")
        print("LLM JUDGE VALIDATION (validate-only mode)")
        print(f"{'='*80}")
        print(f"Input parquet: {input_path}")
        print(f"PDF directory: {args.pdf_dir}")
        print(f"Output: {output_path}")
        print(f"{'='*80}")
        
        judge = LLMJudge(provider=args.judge_provider, model=args.judge_model)
        validate_existing_parquet(input_path, output_path, args.pdf_dir, judge, dpi=args.dpi)
        print("\nDone!")
        return
    
    # Get list of PDF files to process
    pdf_files = []
    base_dir = None  # Base directory for relative path calculation
    if os.path.isfile(input_path):
        pdf_files = [input_path]
        base_dir = os.path.dirname(input_path) or "."
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}_payment_info.parquet"
    elif os.path.isdir(input_path):
        pdf_files = glob.glob(os.path.join(input_path, "*.pdf"))
        pdf_files.sort()
        base_dir = input_path
        if not pdf_files:
            print(f"Error: No PDF files found in {input_path}")
            sys.exit(1)
        if output_path is None:
            dir_name = os.path.basename(input_path.rstrip('/'))
            output_path = f"{dir_name}_payment_info.parquet"
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)
    
    print(f"{'='*80}")
    print("GEMINI 2.5 PRO TAX PAYMENT EXTRACTOR")
    if args.validate:
        print(f"  + LLM Judge Validation ({args.judge_provider})")
    print(f"{'='*80}")
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    print(f"Output will be saved to: {output_path}")
    print(f"DPI: {args.dpi}")
    print(f"{'='*80}")
    
    # Initialize Gemini OCR
    ocr = GeminiOCR(model=args.model)
    
    # Initialize judge if validation enabled
    judge = None
    if args.validate:
        judge = LLMJudge(provider=args.judge_provider, model=args.judge_model)
    
    # Process all PDFs
    all_results = []
    successful = 0
    failed = 0
    federal_payments_found = 0
    state_payments_found = 0
    accurate_count = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] ", end='')
        result = process_pdf(pdf_path, ocr, dpi=args.dpi, judge=judge, base_dir=base_dir)
        all_results.append(result)
        
        if result.get('error') and 'Error' in str(result.get('error', '')):
            failed += 1
        else:
            successful += 1
            if result.get('federal_amount_due') is not None:
                federal_payments_found += 1
            state_payments = json.loads(result.get('state_payments', '[]'))
            state_payments_found += len([sp for sp in state_payments if sp.get('amount_due') is not None])
            if args.validate and result.get('is_accurate'):
                accurate_count += 1
    
    # Save results to parquet
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    df = pd.DataFrame(all_results)
    df.to_parquet(output_path, index=False)
    print(f"✓ Successfully saved {len(df)} PDF(s) to {output_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Total PDFs processed: {len(df)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total pages: {df['num_pages'].sum()}")
    print(f"  Federal payments found: {federal_payments_found}")
    print(f"  State payments found: {state_payments_found}")
    
    if args.validate:
        print(f"\n  VALIDATION:")
        print(f"  Accurate: {accurate_count}/{successful} ({accurate_count/max(successful,1)*100:.1f}%)")
        print(f"  Issues found: {successful - accurate_count}")
        if 'confidence' in df.columns:
            print(f"  Average confidence: {df['confidence'].mean():.1%}")
            print(f"  Low confidence (<70%): {(df['confidence'] < 0.7).sum()}/{len(df)}")
    
    print(f"{'='*80}")
    print("Done!")


if __name__ == "__main__":
    main()

