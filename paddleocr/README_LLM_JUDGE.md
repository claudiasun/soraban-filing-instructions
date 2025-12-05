# LLM Judge for OCR Validation

This tool uses Large Language Models (LLMs) with vision capabilities to validate money amounts extracted from PDF documents using PaddleOCR.

## Features

- **Automated Validation**: Uses GPT-4o or Claude 3.5 Sonnet to verify extracted money amounts
- **Multiple Validation Metrics**: 
  - `is_aligned`: Whether extracted values match what's in the image
  - `is_reasonable`: Whether values are reasonable for tax/financial documents
  - `note`: Detailed explanation of any issues found
  - `confidence`: Score (0-1) indicating if human review is needed
  - `value_bbox`: Bounding boxes of money amounts for easy verification
- **Comprehensive Analysis**: LLM identifies amounts that were:
  - Correctly extracted
  - Missing from extraction
  - Incorrectly extracted

## Installation

```bash
# Install required packages
pip install pandas pyarrow pdf2image pillow

# Install LLM provider (choose one or both)
pip install openai          # For GPT-4o
pip install anthropic       # For Claude

# PaddleOCR (optional, for bounding box extraction)
pip install paddleocr paddlepaddle
```

### System Dependencies

For PDF processing, you need poppler:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

## Setup

### Option 1: OpenAI (GPT-4o)

1. Get an API key from https://platform.openai.com/api-keys
2. Set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Option 2: Anthropic (Claude)

1. Get an API key from https://console.anthropic.com/
2. Set environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Basic Usage

```bash
# Validate with OpenAI (default)
python llm_judge.py input.parquet output.parquet

# Validate with Anthropic Claude
python llm_judge.py input.parquet output.parquet --provider anthropic
```

### With PDF Directory

If your PDFs are in a specific directory:

```bash
python llm_judge.py results.parquet validated.parquet --pdf-dir "Filing instructions"
```

### Examples

```bash
# Example 1: Validate single PDF results
python llm_judge.py "CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet" \
                   validated_results.parquet \
                   --pdf-dir "Filing instructions"

# Example 2: Validate with Claude (often better for document analysis)
python llm_judge.py "Filing instructions/Filing instructions_ocr_results.parquet" \
                   "Filing instructions/Filing instructions_validated.parquet" \
                   --provider anthropic

# Example 3: Use specific model
python llm_judge.py results.parquet validated.parquet --provider openai --model gpt-4o
```

## Output Columns

The script adds the following columns to your parquet file:

| Column | Type | Description |
|--------|------|-------------|
| `is_aligned` | bool | True if extracted amounts match the image |
| `is_reasonable` | bool | True if amounts are reasonable for the document type |
| `note` | str | Explanation if misaligned or unreasonable |
| `confidence` | float | 0-1 score (lower means needs human review) |
| `value_bbox` | str | JSON string of bounding boxes for money amounts |
| `amounts_found_by_llm` | str | All amounts the LLM identified in the image |
| `missing_amounts` | str | Amounts visible but not extracted by OCR |
| `incorrect_amounts` | str | Amounts extracted incorrectly |

## Understanding Confidence Scores

- **0.9 - 1.0**: High confidence, likely correct
- **0.7 - 0.9**: Medium confidence, spot check recommended
- **0.0 - 0.7**: Low confidence, human review needed

## Example Workflow

```python
import pandas as pd

# 1. Run OCR on PDFs
# (using your existing run.py script)
# python run.py "Filing instructions" results.parquet

# 2. Validate with LLM
# python llm_judge.py results.parquet validated.parquet --provider anthropic

# 3. Analyze results
df = pd.read_parquet('validated.parquet')

# Find documents needing review
needs_review = df[df['confidence'] < 0.7]
print(f"Documents needing review: {len(needs_review)}")

# Find misaligned extractions
misaligned = df[~df['is_aligned']]
print(f"Misaligned extractions: {len(misaligned)}")

# View details
for idx, row in needs_review.iterrows():
    print(f"\n{row['pdf_url']}:")
    print(f"  Extracted: {row['money_amounts']}")
    print(f"  LLM found: {row['amounts_found_by_llm']}")
    print(f"  Missing: {row['missing_amounts']}")
    print(f"  Note: {row['note']}")
```

## Cost Considerations

**OpenAI GPT-4o:**
- ~$0.00025 per page (image input)
- ~$0.01 per request (text output)

**Anthropic Claude 3.5 Sonnet:**
- ~$0.003 per page (image input)
- ~$0.015 per request (text output)

For a 100-page document batch:
- OpenAI: ~$1.25 total
- Anthropic: ~$1.80 total

## Troubleshooting

### "PDF file not found"
The script looks for PDFs in:
1. Current directory
2. `Filing instructions/` subdirectory
3. Parent directory

Use `--pdf-dir` to specify exact location.

### "ModuleNotFoundError: No module named 'openai'"
Install the required provider:
```bash
pip install openai  # or: pip install anthropic
```

### "Error initializing OpenAI client"
Make sure your API key is set:
```bash
echo $OPENAI_API_KEY  # Should print your key
# If not set:
export OPENAI_API_KEY="sk-..."
```

### Memory issues with large PDFs
The script loads entire PDFs into memory. For very large documents:
1. Process them individually
2. Reduce DPI (currently 200)
3. Process page-by-page instead of all at once

## Advanced Usage

### Programmatic Use

```python
from llm_judge import LLMJudge, validate_parquet

# Option 1: Use the full validation function
validate_parquet(
    'input.parquet',
    'output.parquet',
    provider='anthropic',
    pdf_directory='Filing instructions'
)

# Option 2: Use LLM judge directly
from pdf2image import convert_from_path

judge = LLMJudge(provider='openai', model='gpt-4o')
images = convert_from_path('document.pdf')

result = judge.call_llm_with_vision(
    images[0],
    extracted_amounts='$1,234.56, $789.00',
    full_text='Full OCR text here...'
)

print(result['is_aligned'])
print(result['note'])
```

### Batch Processing Multiple Files

```bash
# Create a script to process all parquet files
for file in *.parquet; do
    echo "Processing $file..."
    python llm_judge.py "$file" "${file%.parquet}_validated.parquet" --provider anthropic
done
```

## Tips for Best Results

1. **Use high-quality PDFs**: Better image quality = better LLM analysis
2. **Check confidence scores**: Always review items with confidence < 0.7
3. **Compare both providers**: GPT-4o and Claude may catch different issues
4. **Verify bounding boxes**: Use the `value_bbox` field to visually confirm locations
5. **Review missing amounts**: The LLM often finds amounts OCR missed

## License

MIT

