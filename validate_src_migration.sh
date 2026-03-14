#!/bin/bash
# Validation script for src directory migration
# This script verifies that the src submodule has been successfully removed
# and replaced with regular Python source files

echo "=========================================================================="
echo "MedRAG Source Directory Validation"
echo "=========================================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# Test 1: Check for .gitmodules file
echo -n "1. Checking for .gitmodules file... "
if [ ! -f .gitmodules ]; then
    echo -e "${GREEN}✓ PASS${NC} (No .gitmodules file)"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} (.gitmodules exists)"
    ((FAIL++))
fi

# Test 2: Check git submodule status
echo -n "2. Checking git submodule status... "
SUBMODULE_OUTPUT=$(git submodule status 2>&1)
if [ -z "$SUBMODULE_OUTPUT" ]; then
    echo -e "${GREEN}✓ PASS${NC} (No submodules configured)"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} (Submodules found)"
    ((FAIL++))
fi

# Test 3: Check for mode 160000 entries (submodule references)
echo -n "3. Checking for submodule entries in git index... "
SUBMODULE_ENTRIES=$(git ls-files --stage | grep "^160000" || echo "")
if [ -z "$SUBMODULE_ENTRIES" ]; then
    echo -e "${GREEN}✓ PASS${NC} (No submodule entries)"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} (Submodule entries found)"
    ((FAIL++))
fi

# Test 4: Check src directory exists
echo -n "4. Checking src directory exists... "
if [ -d "src" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAIL++))
fi

# Test 5: Check for required Python files
echo -n "5. Checking for required Python files... "
REQUIRED_FILES=(
    "src/__init__.py"
    "src/model_registry.py"
    "src/ledger.py"
    "src/inference.py"
    "src/models.py"
    "src/demo_rag_vfl.py"
    "src/Aggregator.sol"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} (All required files present)"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} (Missing files: ${MISSING_FILES[*]})"
    ((FAIL++))
fi

# Test 6: Check Python syntax
echo -n "6. Checking Python syntax... "
SYNTAX_OK=true
for pyfile in src/*.py; do
    if [ -f "$pyfile" ]; then
        if ! python -m py_compile "$pyfile" 2>/dev/null; then
            SYNTAX_OK=false
            break
        fi
    fi
done

if $SYNTAX_OK; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} (Syntax errors found)"
    ((FAIL++))
fi

# Test 7: Check git tracking
echo -n "7. Checking git tracking of src files... "
TRACKED_FILES=$(git ls-files src/ | wc -l)
if [ "$TRACKED_FILES" -gt 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} ($TRACKED_FILES files tracked)"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} (No files tracked in src/)"
    ((FAIL++))
fi

# Test 8: Check for documentation
echo -n "8. Checking for documentation... "
if [ -f "src/README.md" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (No README in src/)"
    ((PASS++))  # Don't fail on this
fi

# Summary
echo ""
echo "=========================================================================="
echo "Validation Summary"
echo "=========================================================================="
echo -e "Tests Passed: ${GREEN}$PASS${NC}"
echo -e "Tests Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All validation tests passed!${NC}"
    echo ""
    echo "The src directory has been successfully migrated from a submodule"
    echo "to a regular directory with Python source files."
    echo ""
    echo "Next steps:"
    echo "  1. Review the created files in src/"
    echo "  2. Implement any missing functionality"
    echo "  3. Run: python -m pytest tests/"
    echo "  4. Test the demo: python src/demo_rag_vfl.py --help"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some validation tests failed!${NC}"
    echo ""
    echo "Please review the failures above and fix them before proceeding."
    echo ""
    exit 1
fi
