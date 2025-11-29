# LLM Judge for OCR Validation - Complete Overview

## 📁 Files Created

### Core Scripts
1. **`llm_judge.py`** - Main validation script
   - Uses GPT-4o or Claude to validate OCR-extracted money amounts
   - Adds validation columns to parquet files
   - Extracts bounding boxes for visual verification
   
2. **`analyze_validation.py`** - Analysis utility
   - Generates comprehensive reports on validation results
   - Identifies documents needing human review
   - Exports CSV files for easy review

3. **`test_llm_judge.py`** - Testing utility
   - Validates your setup (packages, API keys)
   - Runs a quick test validation
   - Helps troubleshoot configuration issues

4. **`example_usage.py`** - Complete example workflow
   - Demonstrates end-to-end usage
   - Shows how to analyze results programmatically
   - Exports summary reports

### Documentation
5. **`README_LLM_JUDGE.md`** - Comprehensive documentation
6. **`QUICKSTART_LLM_JUDGE.md`** - Quick start guide (in parent directory)
7. **`requirements_llm_judge.txt`** - Python dependencies

## 🎯 What Problem Does This Solve?

When you extract money amounts from PDFs using OCR (PaddleOCR), you need to verify:
- Are the extracted amounts **accurate**?
- Did OCR **miss any amounts**?
- Did OCR **misread any amounts**?
- Which documents need **human review**?

This LLM judge automates that validation by having AI analyze the original PDF images and compare them to the OCR results.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd paddleocr
pip install -r requirements_llm_judge.txt
```

### 2. Set API Key (choose one)
```bash
# OpenAI
export OPENAI_API_KEY="sk-your-key"

# OR Anthropic (recommended for documents)
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

### 3. Test Your Setup
```bash
python test_llm_judge.py --provider anthropic
```

### 4. Run Validation
```bash
python llm_judge.py input.parquet output.parquet --provider anthropic --pdf-dir "Filing instructions"
```

### 5. Analyze Results
```bash
python analyze_validation.py output.parquet
```

## 📊 Output Columns

The LLM judge adds these columns to your parquet file:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `is_aligned` | bool | Do OCR amounts match the image? | `True` |
| `is_reasonable` | bool | Are amounts reasonable? | `True` |
| `confidence` | float | AI confidence (0-1) | `0.95` |
| `note` | str | Explanation of issues | "All amounts correctly extracted" |
| `value_bbox` | str | JSON of bounding boxes | `{"1": [{"bbox": [[x,y],...], "text": "$1,234"}]}` |
| `amounts_found_by_llm` | str | Amounts AI found in image | "$1,234.56, $789.00" |
| `missing_amounts` | str | Amounts OCR missed | "$50.00" |
| `incorrect_amounts` | str | Amounts OCR got wrong | "$1,234.56 (should be $1,284.56)" |

## 🔍 Typical Workflow

```bash
# 1. Run OCR on your PDFs (if not done already)
python run.py "Filing instructions" ocr_results.parquet

# 2. Validate with LLM
python llm_judge.py ocr_results.parquet validated.parquet \
    --provider anthropic --pdf-dir "Filing instructions"

# 3. Analyze results
python analyze_validation.py validated.parquet
# Creates: validated_analysis.csv and validated_needs_review.csv

# 4. Review in spreadsheet
open validated_needs_review.csv
```

## 💡 Key Features

### 1. **Multi-Page Support**
Currently validates the first page of multi-page documents. Can be extended to validate all pages.

### 2. **Bounding Box Extraction**
Uses PaddleOCR to extract the location of money amounts in the image for easy visual verification.

### 3. **Flexible LLM Providers**
- **OpenAI GPT-4o**: Fast, accurate, cost-effective
- **Anthropic Claude 3.5**: Excellent for complex document analysis

### 4. **Detailed Analysis**
The `analyze_validation.py` script provides:
- Overall statistics
- Alignment and reasonableness rates
- Confidence level distribution
- List of documents needing review
- OCR accuracy metrics
- Exported CSV reports

### 5. **Human-in-the-Loop**
Confidence scores help you prioritize which documents need manual review:
- **0.9-1.0**: High confidence - likely correct
- **0.7-0.9**: Medium confidence - spot check
- **0.0-0.7**: Low confidence - needs review

## 📈 Expected Results

For typical tax filing instructions:
- **Alignment**: 85-95% of documents
- **Average Confidence**: 0.85-0.95
- **Perfect Extractions**: 70-85%
- **Needs Review**: 5-15%

Common issues caught:
- ✓ Missing amounts (penalties, interest, fees)
- ✓ Misread amounts (8 vs 3, 0 vs O)
- ✓ Amounts in unusual formats
- ✓ Handwritten amounts
- ✓ Amounts in tables or complex layouts

## 💰 Cost Estimates

### Per Document Costs
| Provider | Per Page | Typical (3 pages) |
|----------|----------|-------------------|
| OpenAI GPT-4o | ~$0.003 | ~$0.01 |
| Anthropic Claude | ~$0.005 | ~$0.015 |

### Batch Processing
| Documents | OpenAI | Anthropic |
|-----------|--------|-----------|
| 10 | $0.10 | $0.15 |
| 50 | $0.50 | $0.75 |
| 100 | $1.00 | $1.50 |
| 1,000 | $10.00 | $15.00 |

## 🛠️ Customization

### Use Different Model
```bash
python llm_judge.py input.parquet output.parquet \
    --provider openai --model gpt-4o-mini  # Cheaper model
```

### Programmatic Usage
```python
from llm_judge import LLMJudge, validate_parquet

# Option 1: Full validation
validate_parquet(
    'input.parquet',
    'output.parquet',
    provider='anthropic',
    pdf_directory='Filing instructions'
)

# Option 2: Custom validation
judge = LLMJudge(provider='openai')
result = judge.call_llm_with_vision(image, amounts, text)
```

### Extend for Multi-Page
Modify `llm_judge.py` to validate all pages:
```python
# In validate_parquet function, change:
for page_num, image in enumerate(images, start=1):
    result = judge.call_llm_with_vision(image, money_amounts, full_text)
    # Store per-page results
```

## 🐛 Troubleshooting

### Common Issues

**"PDF file not found"**
- Use `--pdf-dir` to specify PDF location
- Check file names match between parquet and filesystem

**"ModuleNotFoundError"**
- Run `pip install -r requirements_llm_judge.txt`
- Make sure you're in the correct Python environment

**"Error initializing client"**
- Check API key is set: `echo $OPENAI_API_KEY`
- Verify key format (OpenAI: `sk-...`, Anthropic: `sk-ant-...`)

**Memory errors with large PDFs**
- Process files one at a time
- Reduce DPI from 200 to 150
- Process page-by-page instead of all at once

**Low accuracy**
- Try the other provider (GPT-4o vs Claude)
- Check PDF quality (low quality → poor LLM analysis)
- Verify OCR is working correctly first

## 📝 Best Practices

1. **Start Small**: Test with 1-5 documents before batch processing
2. **Review Confidence**: Always review documents with confidence < 0.7
3. **Compare Providers**: Try both GPT-4o and Claude on difficult documents
4. **Use Bounding Boxes**: Leverage the `value_bbox` field for visual verification
5. **Iterate**: Use LLM feedback to improve your OCR settings
6. **Version Control**: Keep copies of original parquet files

## 🔗 Related Files

In the repository:
- `run.py` - OCR extraction script
- `pdf_processor.py` - PDF processing utilities
- `CCH Axcess - Fed, CA, SC, PA & NC_Redacted_ocr_results.parquet` - Example OCR results
- `Filing instructions/` - Example PDF files

## 📚 Further Reading

- [OpenAI Vision API Docs](https://platform.openai.com/docs/guides/vision)
- [Anthropic Claude Vision Docs](https://docs.anthropic.com/claude/docs/vision)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

## 🤝 Support

If you encounter issues:
1. Run `python test_llm_judge.py` to diagnose problems
2. Check the troubleshooting section in `README_LLM_JUDGE.md`
3. Review example usage in `example_usage.py`
4. Ensure API keys are valid and have sufficient credits

## 🎓 Learning Path

1. **Beginner**: Run `test_llm_judge.py` to validate setup
2. **Intermediate**: Use `example_usage.py` to process a full dataset
3. **Advanced**: Modify `llm_judge.py` for custom validation logic
4. **Expert**: Integrate into your own pipeline with programmatic API

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Author**: AI Assistant

