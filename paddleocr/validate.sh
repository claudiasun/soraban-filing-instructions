#!/bin/bash
#
# Quick validation script for LLM judge
#
# Usage:
#   ./validate.sh input.parquet [provider]
#   ./validate.sh input.parquet anthropic
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.parquet> [provider]"
    echo ""
    echo "Examples:"
    echo "  $0 results.parquet"
    echo "  $0 results.parquet anthropic"
    echo "  $0 results.parquet openai"
    exit 1
fi

INPUT_FILE="$1"
PROVIDER="${2:-anthropic}"  # Default to anthropic
OUTPUT_FILE="${INPUT_FILE%.parquet}_validated.parquet"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Check API key based on provider
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
        echo "Set it with: export OPENAI_API_KEY='sk-...'"
        exit 1
    fi
    echo -e "${GREEN}✓ OpenAI API key found${NC}"
elif [ "$PROVIDER" = "anthropic" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${RED}Error: ANTHROPIC_API_KEY not set${NC}"
        echo "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        exit 1
    fi
    echo -e "${GREEN}✓ Anthropic API key found${NC}"
else
    echo -e "${RED}Error: Unknown provider: $PROVIDER${NC}"
    echo "Use 'openai' or 'anthropic'"
    exit 1
fi

echo ""
echo "=========================================="
echo "LLM Judge Validation"
echo "=========================================="
echo "Input:    $INPUT_FILE"
echo "Output:   $OUTPUT_FILE"
echo "Provider: $PROVIDER"
echo "=========================================="
echo ""

# Run validation
echo -e "${YELLOW}Running validation...${NC}"
python3 llm_judge.py "$INPUT_FILE" "$OUTPUT_FILE" \
    --provider "$PROVIDER" \
    --pdf-dir "Filing instructions"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Validation complete!${NC}"
    echo ""
    
    # Run analysis
    echo -e "${YELLOW}Generating analysis report...${NC}"
    python3 analyze_validation.py "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Analysis complete!${NC}"
        echo ""
        echo "Generated file:"
        echo "  - $OUTPUT_FILE (validated parquet)"
        echo ""
        echo -e "${GREEN}Done!${NC}"
    else
        echo -e "${YELLOW}Warning: Analysis failed${NC}"
    fi
else
    echo -e "${RED}Error: Validation failed${NC}"
    exit 1
fi

