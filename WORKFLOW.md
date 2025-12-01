# OCR Validation Workflow

## 📊 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT: PDF FILES                         │
│    (Tax Filing Instructions, Financial Documents, etc.)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: OCR EXTRACTION                        │
│                     (run.py with PaddleOCR)                      │
│                                                                   │
│  • Convert PDF to images (200 DPI)                              │
│  • Run PaddleOCR on each page                                   │
│  • Extract money amounts using regex patterns                   │
│  • Combine all pages per document                               │
│                                                                   │
│  Pattern 1: Explicit $: \$\d+(?:,\d{3})*(?:\.\d{2})?          │
│  Pattern 2: Context: (refund|payment|...) \$?\d+...            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT: OCR PARQUET FILE                       │
│                                                                   │
│  Columns:                                                        │
│  • pdf_url: "Drake - Penalty_Redacted.pdf"                      │
│  • num_pages: 3                                                  │
│  • money_amounts: "$1,234.56, $789.00"                          │
│  • full_text: "Payment Due\nAmount: $1,234.56\n..."            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 2: AI VALIDATION                          │
│            (llm_judge.py with GPT-4o or Claude)                 │
│                                                                   │
│  For each document:                                             │
│  1. Find original PDF file                                      │
│  2. Convert PDF to images (200 DPI)                            │
│  3. Extract bounding boxes with PaddleOCR                      │
│  4. Send to LLM with prompt:                                   │
│     "Here's an image and OCR-extracted amounts.                │
│      Verify: Are they aligned? Reasonable?                     │
│      What amounts do you see? What's missing?"                 │
│  5. LLM returns structured JSON response                       │
│  6. Add validation columns to dataframe                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: VALIDATED PARQUET FILE                      │
│                                                                   │
│  Original columns +                                             │
│  • is_aligned: True                                             │
│  • is_reasonable: True                                          │
│  • confidence: 0.95                                             │
│  • note: "All amounts correctly extracted"                     │
│  • value_bbox: {"1": [{"bbox": [[x,y]...], "text": "$1,234"}]} │
│  • amounts_found_by_llm: "$1,234.56, $789.00"                  │
│  • missing_amounts: ""                                          │
│  • incorrect_amounts: ""                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 3: ANALYSIS & REPORTING                   │
│                   (analyze_validation.py)                       │
│                                                                   │
│  • Calculate overall statistics                                │
│  • Identify documents needing review                           │
│  • Generate CSV reports:                                       │
│    - Full analysis with all columns                           │
│    - Filtered list of docs needing review                     │
│  • Print summary to console                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT: ANALYSIS REPORTS                       │
│                                                                   │
│  1. validated_analysis.csv                                      │
│     → All documents with validation details                     │
│                                                                   │
│  2. validated_needs_review.csv                                  │
│     → Filtered list: confidence < 0.7 or not aligned          │
│                                                                   │
│  3. Console summary:                                            │
│     ✓ 12/15 aligned                                            │
│     ✓ 14/15 reasonable                                         │
│     ⚠️  3/15 need review                                        │
│     📊 0.87 avg confidence                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 4: HUMAN REVIEW                           │
│                  (Manual verification)                          │
│                                                                   │
│  Review documents where:                                        │
│  • confidence < 0.7                                             │
│  • is_aligned = False                                           │
│  • missing_amounts has values                                   │
│  • incorrect_amounts has values                                 │
│                                                                   │
│  Use value_bbox to locate amounts in original PDF              │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

```
PDF → Images → OCR Text → Money Amounts → Validation → Report
 ↓       ↓        ↓            ↓              ↓           ↓
File   PNG    String      "$1,234.56"    True/False    CSV
```

## 🎯 Decision Tree: When to Use Each Tool

```
Start Here
    │
    ├─→ Need to extract money amounts from PDFs?
    │   └─→ YES → Use run.py
    │       └─→ Creates: results.parquet
    │
    ├─→ Have OCR results, need to verify accuracy?
    │   └─→ YES → Use llm_judge.py
    │       └─→ Creates: validated.parquet
    │
    ├─→ Have validated results, want detailed analysis?
    │   └─→ YES → Use analyze_validation.py
    │       └─→ Creates: analysis.csv, needs_review.csv
    │
    ├─→ Want to test your setup first?
    │   └─→ YES → Use test_llm_judge.py
    │       └─→ Validates environment and runs test
    │
    └─→ Want to do everything in one command?
        └─→ YES → Use validate.sh
            └─→ Runs llm_judge.py + analyze_validation.py
```

## 📈 Validation Logic Flow

```python
for each_document in parquet_file:
    
    # 1. Load original PDF
    pdf = find_pdf(document.pdf_url)
    if not pdf:
        mark_as_error("PDF not found")
        continue
    
    # 2. Convert to images
    images = convert_pdf_to_images(pdf, dpi=200)
    
    # 3. Extract bounding boxes (optional)
    bboxes = extract_money_bboxes_with_ocr(images)
    
    # 4. Call LLM for validation
    llm_response = llm.analyze({
        "image": images[0],  # First page
        "ocr_amounts": document.money_amounts,
        "ocr_text": document.full_text,
        "prompt": "Validate these money amounts..."
    })
    
    # 5. Parse LLM response
    validation = {
        "is_aligned": llm_response.is_aligned,
        "is_reasonable": llm_response.is_reasonable,
        "confidence": llm_response.confidence,
        "note": llm_response.note,
        "amounts_found_by_llm": llm_response.amounts_found,
        "missing_amounts": llm_response.missing_amounts,
        "incorrect_amounts": llm_response.incorrect_amounts,
        "value_bbox": bboxes
    }
    
    # 6. Add to dataframe
    add_validation_columns(document, validation)
```

## 🔍 Confidence Score Calculation

The LLM determines confidence based on:

```
High Confidence (0.9-1.0):
  ✓ All amounts clearly visible
  ✓ OCR matches exactly
  ✓ Standard formatting
  ✓ No ambiguity

Medium Confidence (0.7-0.9):
  ⚠ Most amounts match
  ⚠ Minor formatting differences
  ⚠ Some amounts in unusual locations

Low Confidence (0.0-0.7):
  ❌ Amounts don't match
  ❌ Missing amounts visible in image
  ❌ OCR misread amounts
  ❌ Handwritten or poor quality
  ❌ Complex layout/tables
```

## 📊 Example: Processing 100 Documents

```
Time Estimates:
┌──────────────────────┬──────────┬───────────────┐
│ Step                 │ Time     │ Cumulative    │
├──────────────────────┼──────────┼───────────────┤
│ OCR Extraction       │ ~10 min  │ 10 min        │
│ AI Validation        │ ~30 min  │ 40 min        │
│ Analysis             │ ~30 sec  │ 40.5 min      │
│ Human Review (3 docs)│ ~15 min  │ 55.5 min      │
└──────────────────────┴──────────┴───────────────┘

Cost Estimates (Anthropic):
┌──────────────────────┬──────────┐
│ Component            │ Cost     │
├──────────────────────┼──────────┤
│ OCR (PaddleOCR)      │ Free     │
│ AI Validation        │ ~$1.50   │
│ Analysis             │ Free     │
├──────────────────────┼──────────┤
│ Total                │ ~$1.50   │
└──────────────────────┴──────────┘
```

## 🎨 Visual: Bounding Box Structure

```json
{
  "1": [  // Page 1
    {
      "bbox": [[120, 450], [280, 450], [280, 480], [120, 480]],
      "text": "$1,234.56",
      "confidence": 0.98,
      "page": 1
    },
    {
      "bbox": [[120, 520], [260, 520], [260, 550], [120, 550]],
      "text": "Payment: $789.00",
      "confidence": 0.95,
      "page": 1
    }
  ],
  "2": [  // Page 2
    {
      "bbox": [[100, 300], [200, 300], [200, 330], [100, 330]],
      "text": "$50.00",
      "confidence": 0.92,
      "page": 2
    }
  ]
}
```

Use these coordinates to:
- Draw rectangles on original image
- Crop specific amounts for review
- Verify OCR accuracy visually

## 🚦 Status Indicators

```
✅ Perfect Extraction:
   • is_aligned = True
   • is_reasonable = True
   • confidence >= 0.9
   • no missing amounts
   • no incorrect amounts
   → No action needed

⚠️  Needs Spot Check:
   • is_aligned = True
   • confidence 0.7-0.9
   → Quick visual verification recommended

❌ Needs Review:
   • is_aligned = False OR
   • confidence < 0.7 OR
   • has missing/incorrect amounts
   → Manual review required
```

---

**Next Steps**: See [QUICKSTART_LLM_JUDGE.md](QUICKSTART_LLM_JUDGE.md) to get started!

