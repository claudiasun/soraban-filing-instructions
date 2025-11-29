# Tax Filing Instructions OCR Validation

This repository contains tools for extracting and validating money amounts from tax filing instruction PDFs using OCR and AI.

## 🎯 Overview

The workflow consists of two main steps:

1. **OCR Extraction** - Extract text and money amounts from PDFs using PaddleOCR
2. **AI Validation** - Validate extracted amounts using LLM vision models (GPT-4o or Claude)

## 📁 Repository Structure

```
.
├── Filing instructions/          # PDF files to process
│   ├── CCH Axcess - *.pdf
│   ├── Drake - *.pdf
│   ├── Lacerte - *.pdf
│   ├── Proconnect - *.pdf
│   └── Ultratax - *.pdf
│
├── paddleocr/                    # OCR and validation tools
│   ├── run.py                    # OCR extraction script
│   ├── llm_judge.py             # AI validation script ⭐
│   ├── analyze_validation.py    # Analysis and reporting
│   ├── test_llm_judge.py        # Setup testing
│   ├── example_usage.py         # Complete example
│   ├── validate.sh              # One-command validation
│   └── README_LLM_JUDGE.md      # Detailed documentation
│
├── QUICKSTART_LLM_JUDGE.md      # Quick start guide
└── README.md                    # This file
```

## 🚀 Quick Start

### Step 1: Extract Money Amounts with OCR

```bash
cd paddleocr

# Process a single PDF
python run.py "Filing instructions/CCH Axcess - Fed, CA, SC, PA & NC_Redacted.pdf"

# Or process all PDFs in a directory
python run.py "Filing instructions" filing_instructions_ocr.parquet
```

### Step 2: Validate with AI

```bash
# Install dependencies
pip install -r requirements_llm_judge.txt

# Set API key (choose one)
export ANTHROPIC_API_KEY="sk-ant-..."  # Recommended
# OR
export OPENAI_API_KEY="sk-..."

# Validate the extraction
python llm_judge.py filing_instructions_ocr.parquet \
                   filing_instructions_validated.parquet \
                   --provider anthropic \
                   --pdf-dir "Filing instructions"

# Analyze results
python analyze_validation.py filing_instructions_validated.parquet
```

### Alternative: One-Command Validation

```bash
./validate.sh filing_instructions_ocr.parquet anthropic
```

## 📊 What You Get

### OCR Output Columns
- `pdf_url` - Filename
- `num_pages` - Number of pages
- `money_amounts` - Extracted amounts (comma-separated)
- `text_preview` - First 500 chars of text
- `full_text` - Complete OCR text

### AI Validation Columns (Added by LLM Judge)
- ✅ `is_aligned` - Do extracted amounts match the image?
- ✅ `is_reasonable` - Are amounts reasonable for tax documents?
- 🎯 `confidence` - AI confidence score (0-1)
- 📝 `note` - Detailed explanation
- 📍 `value_bbox` - Bounding boxes for visual verification
- 💰 `amounts_found_by_llm` - All amounts AI identified
- ⚠️ `missing_amounts` - Amounts OCR missed
- ❌ `incorrect_amounts` - Amounts OCR got wrong

## 💡 Example Results

After validation, you'll see which documents need review:

```
Documents needing review: 3 / 15

1. Drake - Penalty and interest but not in filling instructions_Redacted.pdf
   Extracted: $1,234.56, $789.00
   LLM found: $1,234.56, $789.00, $50.00
   ⚠️  Missing: $50.00 (penalty amount in fine print)
   Confidence: 0.65

2. CCH Axcess - Late Payment penalty not listed_Redacted.pdf
   Extracted: $2,500.00
   LLM found: $2,800.00
   ❌ Incorrect: $2,500.00 (should be $2,800.00)
   Confidence: 0.45
```

## 🛠️ Use Cases

### 1. Quality Assurance
Automatically verify that OCR correctly extracted all financial amounts.

### 2. Find Missing Data
Identify money amounts that OCR missed (penalties, interest, fees in fine print).

### 3. Catch Errors
Detect when OCR misread amounts (8 vs 3, 0 vs O, etc.).

### 4. Prioritize Review
Confidence scores help you focus on documents most likely to have issues.

### 5. Improve OCR
Use LLM feedback to identify patterns and improve OCR settings.

## 📈 Expected Accuracy

Based on typical tax filing instructions:
- **OCR Extraction**: 85-95% of amounts correctly extracted
- **AI Validation**: Catches 90-95% of OCR errors
- **False Positives**: <5% (AI thinks there's an error when OCR is correct)

## 💰 Cost

### Per Document (3-page average)
- OCR: Free (PaddleOCR)
- AI Validation with OpenAI: ~$0.01
- AI Validation with Anthropic: ~$0.015

### Batch Processing
| Documents | OpenAI | Anthropic |
|-----------|--------|-----------|
| 10 | $0.10 | $0.15 |
| 100 | $1.00 | $1.50 |
| 1,000 | $10.00 | $15.00 |

## 📚 Documentation

- **[QUICKSTART_LLM_JUDGE.md](QUICKSTART_LLM_JUDGE.md)** - 5-minute setup guide
- **[paddleocr/README_LLM_JUDGE.md](paddleocr/README_LLM_JUDGE.md)** - Comprehensive documentation
- **[paddleocr/LLM_JUDGE_OVERVIEW.md](paddleocr/LLM_JUDGE_OVERVIEW.md)** - Technical overview

## 🧪 Testing Your Setup

Run the test suite to verify everything is configured correctly:

```bash
cd paddleocr
python test_llm_judge.py --provider anthropic
```

This will:
1. ✓ Check all required packages are installed
2. ✓ Verify API keys are configured
3. ✓ Find test files
4. ✓ Run a sample validation
5. ✓ Display results

## 🔧 Requirements

### Python Packages
```bash
pip install pandas pyarrow pillow pdf2image
pip install paddlepaddle paddleocr  # For OCR
pip install anthropic  # For Claude
# OR
pip install openai     # For GPT-4o
```

### System Dependencies
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

## 🎓 Learning Path

1. **Start Here**: Read [QUICKSTART_LLM_JUDGE.md](QUICKSTART_LLM_JUDGE.md)
2. **Test Setup**: Run `python test_llm_judge.py`
3. **Try Example**: Run `python example_usage.py`
4. **Full Workflow**: Process your own PDFs
5. **Dive Deeper**: Read [README_LLM_JUDGE.md](paddleocr/README_LLM_JUDGE.md)

## 📝 Example Workflow

```bash
# 1. Extract with OCR
cd paddleocr
python run.py "../Filing instructions" results.parquet

# 2. Validate with AI
export ANTHROPIC_API_KEY="sk-ant-..."
python llm_judge.py results.parquet validated.parquet --provider anthropic

# 3. Analyze
python analyze_validation.py validated.parquet

# 4. Review in Excel
open validated_needs_review.csv
```

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Submit pull requests
- Share your results

## 📄 License

MIT License - feel free to use and modify for your needs.

## ❓ FAQ

**Q: Which LLM provider should I use?**  
A: Anthropic Claude is generally better for document analysis, but GPT-4o is faster and cheaper. Try both!

**Q: How long does validation take?**  
A: About 10-30 seconds per document, depending on page count and provider.

**Q: Can I validate without re-running OCR?**  
A: Yes! The LLM judge works on existing parquet files from OCR.

**Q: What if the PDF has many pages?**  
A: Currently validates the first page. You can extend the code to validate all pages.

**Q: Is my data sent to OpenAI/Anthropic?**  
A: Yes, PDF images and OCR text are sent for analysis. Don't use with sensitive data unless approved.

**Q: Can I run this offline?**  
A: OCR can run offline, but AI validation requires internet access to LLM APIs.

## 🔗 Resources

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Anthropic Claude Vision](https://docs.anthropic.com/claude/docs/vision)

---

**Need Help?** Run `python test_llm_judge.py` to diagnose issues or check the troubleshooting section in the documentation.

