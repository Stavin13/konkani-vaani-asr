#!/bin/bash
# Project Cleanup Script
# Removes unnecessary files and organizes the project

echo "🧹 Starting Project Cleanup..."
echo "======================================"

# 1. Remove macOS metadata files
echo ""
echo "1️⃣ Removing macOS metadata files (._* files)..."
find . -name "._*" -type f -delete
echo "✅ Removed macOS metadata files"

# 2. Remove Python cache
echo ""
echo "2️⃣ Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
echo "✅ Removed Python cache"

# 3. Move old/unnecessary files to archives
echo ""
echo "3️⃣ Archiving old files..."
mkdir -p archives/old_files

# Move Bengali PDF (not related to Konkani project)
if [ -f "Bengali_NLP_Full_Guide_With_Code.pdf" ]; then
    mv Bengali_NLP_Full_Guide_With_Code.pdf archives/old_files/
    echo "  ✓ Archived Bengali_NLP_Full_Guide_With_Code.pdf"
fi

# Move download.txt (old Kaggle logs)
if [ -f "download.txt" ]; then
    mv download.txt archives/old_files/kaggle_logs_$(date +%Y%m%d).txt
    echo "  ✓ Archived download.txt"
fi

# 4. Consolidate documentation folders
echo ""
echo "4️⃣ Consolidating documentation..."
if [ -d "documentation" ] && [ -d "docs" ]; then
    # Move any unique files from documentation/ to docs/
    rsync -av --ignore-existing documentation/ docs/
    # Remove documentation folder
    rm -rf documentation/
    echo "  ✓ Merged documentation/ into docs/"
fi

# 5. Archive old checkpoints
echo ""
echo "5️⃣ Organizing checkpoints..."
if [ -d "konkanivani_checkpoints_backup" ]; then
    if [ ! -d "archives/checkpoints_backup" ]; then
        mv konkanivani_checkpoints_backup archives/checkpoints_backup
        echo "  ✓ Moved konkanivani_checkpoints_backup to archives/"
    fi
fi

# 6. Clean empty directories
echo ""
echo "6️⃣ Removing empty directories..."
find . -type d -empty -delete 2>/dev/null
echo "✅ Removed empty directories"

# 7. Update .gitignore
echo ""
echo "7️⃣ Updating .gitignore..."
cat >> .gitignore << 'EOF'

# macOS
.DS_Store
._*
.AppleDouble
.LSOverride

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/

# Project specific
download.txt
Bengali_NLP_Full_Guide_With_Code.pdf
konkanivani_checkpoints_backup/

# Large files
*.zip
*.tar.gz
*.pt
*.pth
*.ckpt

# Logs
logs/
*.log

# Outputs
outputs/
checkpoints/*.pt
!checkpoints/.gitkeep
EOF
echo "✅ Updated .gitignore"

# 8. Create .gitkeep files for important empty directories
echo ""
echo "8️⃣ Preserving important directory structure..."
for dir in checkpoints logs outputs; do
    if [ -d "$dir" ]; then
        touch "$dir/.gitkeep"
    fi
done
echo "✅ Added .gitkeep files"

# Summary
echo ""
echo "======================================"
echo "✨ Cleanup Complete!"
echo "======================================"
echo ""
echo "📊 Summary:"
echo "  ✓ Removed macOS metadata files"
echo "  ✓ Removed Python cache"
echo "  ✓ Archived old files"
echo "  ✓ Consolidated documentation"
echo "  ✓ Organized checkpoints"
echo "  ✓ Updated .gitignore"
echo ""
echo "📁 Project structure is now cleaner!"
echo ""
