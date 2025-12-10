#!/bin/bash
# Safe Project Cleanup Script (with preview)
# Shows what will be deleted before actually deleting

echo "🔍 Project Cleanup Preview"
echo "======================================"
echo ""

# Count files to be removed
echo "📊 Files that will be removed:"
echo ""

# macOS metadata files
MACOS_COUNT=$(find . -name "._*" -type f | wc -l | tr -d ' ')
echo "  • macOS metadata files (._*): $MACOS_COUNT files"

# Python cache
PYCACHE_COUNT=$(find . -type d -name "__pycache__" | wc -l | tr -d ' ')
PYC_COUNT=$(find . -type f -name "*.pyc" | wc -l | tr -d ' ')
echo "  • Python cache (__pycache__): $PYCACHE_COUNT directories"
echo "  • Python bytecode (*.pyc): $PYC_COUNT files"

# Old files
echo ""
echo "📦 Files that will be archived:"
[ -f "Bengali_NLP_Full_Guide_With_Code.pdf" ] && echo "  • Bengali_NLP_Full_Guide_With_Code.pdf"
[ -f "download.txt" ] && echo "  • download.txt (Kaggle logs)"
[ -d "konkanivani_checkpoints_backup" ] && echo "  • konkanivani_checkpoints_backup/ folder"

# Documentation
echo ""
echo "📚 Folders that will be merged:"
[ -d "documentation" ] && [ -d "docs" ] && echo "  • documentation/ → docs/"

echo ""
echo "======================================"
echo ""
read -p "❓ Do you want to proceed with cleanup? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🧹 Starting cleanup..."
    bash cleanup_project.sh
else
    echo ""
    echo "❌ Cleanup cancelled."
    echo ""
    echo "💡 To manually clean specific items:"
    echo "   • Remove macOS files: find . -name '._*' -delete"
    echo "   • Remove Python cache: find . -name '__pycache__' -exec rm -rf {} +"
    echo "   • Archive old files: mv Bengali_NLP_Full_Guide_With_Code.pdf archives/"
fi
