# Usage Examples - LLM Judge for OCR Validation

This file contains copy-paste ready examples for common use cases.

## 🚀 Basic Usage

### Example 1: Validate Single File with Anthropic

```bash
cd paddleocr

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Validate
python llm_judge.py \
    "../CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet" \
    "validated_results.parquet" \
    --provider anthropic \
    --pdf-dir "../Filing instructions"

# Analyze results
python analyze_validation.py validated_results.parquet
```

### Example 2: Validate with OpenAI

```bash
cd paddleocr

# Set your API key
export OPENAI_API_KEY="sk-your-key-here"

# Validate
python llm_judge.py \
    "results.parquet" \
    "validated_results.parquet" \
    --provider openai \
    --pdf-dir "../Filing instructions"
```

### Example 3: Use the Shell Script (Easiest)

```bash
cd paddleocr

# Set API key
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Run validation + analysis in one command
./validate.sh results.parquet anthropic
```

## 🧪 Testing

### Test Your Setup

```bash
cd paddleocr

# Test with Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
python test_llm_judge.py --provider anthropic

# Test with OpenAI
export OPENAI_API_KEY="sk-your-key-here"
python test_llm_judge.py --provider openai
```

## 📊 Complete Workflow

### Process PDFs from Scratch

```bash
cd paddleocr

# Step 1: Run OCR on all PDFs in a directory
python run.py "../Filing instructions" filing_instructions_ocr.parquet

# Step 2: Set API key
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Step 3: Validate with LLM
python llm_judge.py \
    filing_instructions_ocr.parquet \
    filing_instructions_validated.parquet \
    --provider anthropic \
    --pdf-dir "../Filing instructions"

# Step 4: Analyze results
python analyze_validation.py filing_instructions_validated.parquet

# Step 5: Open reports in Excel/Numbers
open filing_instructions_validated_needs_review.csv
```

### Process Single PDF

```bash
cd paddleocr

# Step 1: OCR extraction
python run.py \
    "../Filing instructions/Drake - Penalty and interest but not in filling instructions_Redacted.pdf"
# Creates: Drake - Penalty and interest but not in filling instructions_Redacted_ocr_results.parquet

# Step 2: Validation
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
python llm_judge.py \
    "Drake - Penalty and interest but not in filling instructions_Redacted_ocr_results.parquet" \
    "Drake_validated.parquet" \
    --provider anthropic \
    --pdf-dir "../Filing instructions"

# Step 3: Check results
python analyze_validation.py Drake_validated.parquet
```

## 🔍 Analysis Examples

### View Documents Needing Review

```bash
cd paddleocr

# After validation, analyze
python analyze_validation.py validated.parquet

# This creates:
# - validated_analysis.csv (full report)
# - validated_needs_review.csv (filtered)

# Open in spreadsheet
open validated_needs_review.csv
```

### Programmatic Analysis

```python
import pandas as pd

# Load validated results
df = pd.read_parquet('validated.parquet')

# Find documents with low confidence
low_conf = df[df['confidence'] < 0.7]
print(f"Need review: {len(low_conf)}")

# Find documents with missing amounts
has_missing = df[df['missing_amounts'].str.len() > 0]
print(f"Missing amounts: {len(has_missing)}")

# Show details
for idx, row in low_conf.iterrows():
    print(f"\n{row['pdf_url']}")
    print(f"  Extracted: {row['money_amounts']}")
    print(f"  LLM found: {row['amounts_found_by_llm']}")
    print(f"  Note: {row['note']}")
```

## 🛠️ Advanced Usage

### Compare Both Providers

```bash
cd paddleocr

# Validate with both OpenAI and Anthropic
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
python llm_judge.py results.parquet results_openai.parquet --provider openai

# Anthropic
python llm_judge.py results.parquet results_anthropic.parquet --provider anthropic

# Compare results
python -c "
import pandas as pd
df1 = pd.read_parquet('results_openai.parquet')
df2 = pd.read_parquet('results_anthropic.parquet')

print('OpenAI:')
print(f'  Aligned: {df1[\"is_aligned\"].sum()}/{len(df1)}')
print(f'  Avg confidence: {df1[\"confidence\"].mean():.2f}')

print('\nAnthropic:')
print(f'  Aligned: {df2[\"is_aligned\"].sum()}/{len(df2)}')
print(f'  Avg confidence: {df2[\"confidence\"].mean():.2f}')
"
```

### Batch Process Multiple Files

```bash
cd paddleocr

export ANTHROPIC_API_KEY="sk-ant-..."

# Process all parquet files in current directory
for file in ../*.parquet; do
    echo "Processing: $file"
    python llm_judge.py "$file" "${file%.parquet}_validated.parquet" \
        --provider anthropic --pdf-dir "../Filing instructions"
done
```

### Use Specific Model

```bash
cd paddleocr

# Use GPT-4o mini (cheaper)
python llm_judge.py results.parquet validated.parquet \
    --provider openai --model gpt-4o-mini

# Use specific Claude version
python llm_judge.py results.parquet validated.parquet \
    --provider anthropic --model claude-3-5-sonnet-20241022
```

## 🐍 Python API Examples

### Example 1: Validate Single Document

```python
from llm_judge import LLMJudge
from pdf2image import convert_from_path

# Initialize judge
judge = LLMJudge(provider='anthropic')

# Load PDF
images = convert_from_path('document.pdf', dpi=200)

# Validate
result = judge.call_llm_with_vision(
    image=images[0],
    extracted_amounts='$1,234.56, $789.00',
    full_text='Payment Due\nAmount: $1,234.56...'
)

# Check results
print(f"Aligned: {result['is_aligned']}")
print(f"Confidence: {result['confidence']}")
print(f"Note: {result['note']}")
if result['missing_amounts']:
    print(f"Missing: {result['missing_amounts']}")
```

### Example 2: Batch Validation

```python
from llm_judge import validate_parquet

# Validate entire dataset
validate_parquet(
    input_path='ocr_results.parquet',
    output_path='validated_results.parquet',
    provider='anthropic',
    pdf_directory='Filing instructions'
)

print("Validation complete!")
```

### Example 3: Custom Processing

```python
import pandas as pd
from llm_judge import LLMJudge, extract_bounding_boxes
from pdf2image import convert_from_path

# Load OCR results
df = pd.read_parquet('ocr_results.parquet')

# Initialize judge
judge = LLMJudge(provider='openai', model='gpt-4o')

# Process only documents with many amounts
for idx, row in df.iterrows():
    amounts = row['money_amounts'].split(', ')
    
    # Skip if fewer than 3 amounts
    if len(amounts) < 3:
        continue
    
    # Validate this document
    pdf_path = f"Filing instructions/{row['pdf_url']}"
    images = convert_from_path(pdf_path, dpi=200)
    
    result = judge.call_llm_with_vision(
        images[0],
        row['money_amounts'],
        row['full_text']
    )
    
    # Print if not aligned
    if not result['is_aligned']:
        print(f"\n⚠️  {row['pdf_url']}")
        print(f"Extracted: {row['money_amounts']}")
        print(f"LLM found: {result['amounts_found']}")
        print(f"Note: {result['note']}")
```

## 📋 Real-World Scenarios

### Scenario 1: Quality Check Before Production

```bash
# Extract from 5 sample PDFs
cd paddleocr
python run.py sample_pdfs/ sample_ocr.parquet

# Validate with LLM
export ANTHROPIC_API_KEY="sk-ant-..."
python llm_judge.py sample_ocr.parquet sample_validated.parquet \
    --provider anthropic --pdf-dir sample_pdfs

# Check accuracy
python analyze_validation.py sample_validated.parquet

# If >90% aligned, proceed with full batch
# If <90%, adjust OCR settings and retry
```

### Scenario 2: Find All Missing Penalties/Interest

```python
import pandas as pd

df = pd.read_parquet('validated.parquet')

# Filter documents with missing amounts
has_missing = df[df['missing_amounts'].str.len() > 0]

print(f"Documents with missing amounts: {len(has_missing)}\n")

for idx, row in has_missing.iterrows():
    missing = row['missing_amounts']
    
    # Check if it's penalty or interest
    if 'penalty' in row['note'].lower() or 'interest' in row['note'].lower():
        print(f"⚠️  {row['pdf_url']}")
        print(f"   Missing: {missing}")
        print(f"   Note: {row['note']}\n")
```

### Scenario 3: Export for Accountant Review

```python
import pandas as pd

# Load validated results
df = pd.read_parquet('validated.parquet')

# Create review sheet with relevant columns
review_df = df[df['confidence'] < 0.8][[
    'pdf_url',
    'money_amounts',
    'amounts_found_by_llm',
    'missing_amounts',
    'incorrect_amounts',
    'confidence',
    'note'
]].copy()

# Add manual review column
review_df['verified_by'] = ''
review_df['verified_date'] = ''
review_df['correct_amounts'] = ''
review_df['comments'] = ''

# Export to Excel
review_df.to_excel('for_accountant_review.xlsx', index=False)
print("✓ Review sheet created: for_accountant_review.xlsx")
```

## 🔧 Troubleshooting Examples

### Check Installation

```bash
cd paddleocr

# Install requirements
pip install -r requirements_llm_judge.txt

# Verify installation
python -c "
try:
    import pandas; print('✓ pandas')
    import pyarrow; print('✓ pyarrow')
    from PIL import Image; print('✓ pillow')
    from pdf2image import convert_from_path; print('✓ pdf2image')
    print('\n✓ All core packages installed')
except ImportError as e:
    print(f'✗ Missing: {e}')
"
```

### Test API Connection

```bash
cd paddleocr

# Test OpenAI
python -c "
import os
import openai
client = openai.OpenAI()
print('✓ OpenAI client initialized')
print(f'✓ API key: {os.getenv(\"OPENAI_API_KEY\")[:8]}...')
"

# Test Anthropic
python -c "
import os
import anthropic
client = anthropic.Anthropic()
print('✓ Anthropic client initialized')
print(f'✓ API key: {os.getenv(\"ANTHROPIC_API_KEY\")[:12]}...')
"
```

### Debug Validation Issues

```python
# debug_validation.py
import pandas as pd
from llm_judge import LLMJudge
from pdf2image import convert_from_path

# Load single problematic document
df = pd.read_parquet('results.parquet')
row = df[df['pdf_url'] == 'problematic.pdf'].iloc[0]

print("Document:", row['pdf_url'])
print("Extracted amounts:", row['money_amounts'])
print("\nFull text preview:")
print(row['full_text'][:500])

# Try validation
judge = LLMJudge(provider='anthropic')
pdf_path = f"Filing instructions/{row['pdf_url']}"
images = convert_from_path(pdf_path, dpi=200)

result = judge.call_llm_with_vision(
    images[0],
    row['money_amounts'],
    row['full_text']
)

print("\nValidation result:")
print(f"Aligned: {result['is_aligned']}")
print(f"Confidence: {result['confidence']}")
print(f"Amounts found: {result['amounts_found']}")
print(f"Note: {result['note']}")
```

---

## 💡 Tips

1. **Start small**: Test with 1-2 documents before batch processing
2. **Use shell script**: `./validate.sh` is the easiest way for simple cases
3. **Check confidence**: Always review documents with confidence < 0.7
4. **Save API costs**: Use OpenAI for large batches (slightly cheaper)
5. **Better accuracy**: Use Anthropic for complex documents
6. **Iterate**: Use LLM feedback to improve OCR settings

## 🔗 Quick Reference

| Task | Command |
|------|---------|
| Test setup | `python test_llm_judge.py` |
| Validate | `python llm_judge.py input.parquet output.parquet --provider anthropic` |
| Analyze | `python analyze_validation.py output.parquet` |
| One command | `./validate.sh input.parquet anthropic` |
| Full workflow | See "Complete Workflow" section above |

---

For more details, see:
- [QUICKSTART_LLM_JUDGE.md](QUICKSTART_LLM_JUDGE.md) - Quick start guide
- [README_LLM_JUDGE.md](paddleocr/README_LLM_JUDGE.md) - Full documentation
- [WORKFLOW.md](WORKFLOW.md) - Technical workflow diagrams

