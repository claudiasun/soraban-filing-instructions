# Quick Start: LLM Judge for OCR Validation

This guide will help you get started with validating OCR-extracted money amounts using AI.

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
cd paddleocr
pip install -r requirements_llm_judge.txt
```

### 2. Set Up API Key

Choose **one** provider:

**Option A: OpenAI (GPT-4o)**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Option B: Anthropic (Claude)** - Recommended for document analysis
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### 3. Run Validation

```bash
# If you already have OCR results in a parquet file:
python llm_judge.py "CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet" \
                   validated_results.parquet \
                   --provider anthropic \
                   --pdf-dir "Filing instructions"
```

## 📊 What You'll Get

The script adds these validation columns to your parquet file:

```
✓ is_aligned        - Does OCR match the image? (True/False)
✓ is_reasonable     - Are values reasonable? (True/False)  
✓ confidence        - How confident is the AI? (0.0 to 1.0)
✓ note              - Explanation of any issues
✓ value_bbox        - Locations of money amounts in image
✓ amounts_found_by_llm     - All amounts AI found
✓ missing_amounts          - Amounts OCR missed
✓ incorrect_amounts        - Amounts OCR got wrong
```

## 💡 Simple Example

```bash
# 1. Run OCR on your PDFs (if you haven't already)
python run.py "Filing instructions" filing_instructions_ocr.parquet

# 2. Validate the OCR results with AI
python llm_judge.py filing_instructions_ocr.parquet \
                   filing_instructions_validated.parquet \
                   --provider anthropic

# 3. Check the results
python -c "
import pandas as pd
df = pd.read_parquet('filing_instructions_validated.parquet')
print(f'Total docs: {len(df)}')
print(f'Need review: {(df[\"confidence\"] < 0.7).sum()}')
print(f'Misaligned: {(~df[\"is_aligned\"]).sum()}')
"
```

## 🔍 Review Results

Open the CSV summary file in Excel/Numbers:

```bash
# The summary CSV is automatically created
open "filing_instructions_validated_summary.csv"
```

Look for:
- ❌ `is_aligned = False` - OCR didn't match the image
- ⚠️ `confidence < 0.7` - Needs human verification
- 📝 `note` column - AI's explanation

## 📖 Full Documentation

See `paddleocr/README_LLM_JUDGE.md` for:
- Detailed API setup instructions
- Cost estimates
- Programmatic usage
- Troubleshooting

## 💰 Cost Estimate

For typical filing instruction documents:

| Provider | Cost per document | 100 docs |
|----------|------------------|----------|
| OpenAI GPT-4o | ~$0.01 | ~$1.00 |
| Anthropic Claude | ~$0.015 | ~$1.50 |

## ⚡ One-Liner for Common Use Case

```bash
# Validate all PDFs in "Filing instructions" directory
python llm_judge.py \
  "Filing instructions/Filing instructions_ocr_results.parquet" \
  "Filing instructions/Filing instructions_validated.parquet" \
  --provider anthropic \
  --pdf-dir "Filing instructions"
```

## 🆘 Troubleshooting

**"PDF file not found"**
→ Use `--pdf-dir` to specify where PDFs are located

**"ModuleNotFoundError: No module named 'openai'"**
→ Run `pip install openai` or `pip install anthropic`

**"Error initializing OpenAI client"**
→ Check your API key: `echo $OPENAI_API_KEY`

**Out of memory with large PDFs**
→ Process files individually instead of batches

## 🎯 Next Steps

1. Review documents where `confidence < 0.7`
2. Check `missing_amounts` to see what OCR missed
3. Use `value_bbox` to visually verify locations
4. Update your OCR settings if patterns emerge

---

**Questions?** See the full documentation in `paddleocr/README_LLM_JUDGE.md`

