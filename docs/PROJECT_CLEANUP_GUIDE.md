# Project Cleanup Guide

## 🎯 Quick Cleanup

### Option 1: Safe Preview (Recommended)
Shows what will be deleted before doing anything:
```bash
./cleanup_project_safe.sh
```

### Option 2: Direct Cleanup
Cleans immediately without preview:
```bash
./cleanup_project.sh
```

## 🧹 What Gets Cleaned

### 1. macOS Metadata Files (~205,000 files!)
- All `._*` files (created by macOS when copying files)
- `.DS_Store` files
- These are invisible and serve no purpose on non-Mac systems

### 2. Python Cache
- `__pycache__/` directories
- `*.pyc` bytecode files
- `*.egg-info/` directories

### 3. Archived Files
Moved to `archives/old_files/`:
- `Bengali_NLP_Full_Guide_With_Code.pdf` (not related to Konkani)
- `download.txt` (old Kaggle training logs)

### 4. Consolidated Folders
- `documentation/` merged into `docs/`
- `konkanivani_checkpoints_backup/` moved to `archives/`

### 5. Updated .gitignore
Prevents these files from being tracked in git

## 📊 Expected Results

**Before cleanup:**
```
Total files: ~210,000+
Project size: ~5-10 GB
```

**After cleanup:**
```
Total files: ~5,000
Project size: ~2-3 GB (depending on checkpoints)
Cleaner git status
Faster file operations
```

## 🔧 Manual Cleanup Commands

If you prefer to clean specific items manually:

### Remove macOS metadata files
```bash
find . -name "._*" -type f -delete
```

### Remove Python cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Archive old files
```bash
mkdir -p archives/old_files
mv Bengali_NLP_Full_Guide_With_Code.pdf archives/old_files/
mv download.txt archives/old_files/
```

### Merge documentation folders
```bash
rsync -av documentation/ docs/
rm -rf documentation/
```

## 📁 Recommended Project Structure

After cleanup, your project should look like:

```
konkani/
├── archives/              # Old files and backups
│   ├── old_files/        # Archived unnecessary files
│   └── checkpoints_backup/
├── checkpoints/          # Current model checkpoints
├── config/               # Training configurations
├── data/                 # Datasets
├── docs/                 # All documentation (consolidated)
├── konkani/              # Main Python package
├── logs/                 # Training logs
├── models/               # Model definitions
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── training_scripts/     # Training scripts
└── utilities/            # Helper utilities
```

## ⚠️ What's NOT Deleted

The cleanup script preserves:
- All source code (`.py` files)
- Model checkpoints (`.pt` files in checkpoints/)
- Training data
- Configuration files
- Documentation
- Git history

## 🔄 Regular Maintenance

Run cleanup periodically:
```bash
# Weekly or after major changes
./cleanup_project_safe.sh
```

Add to your workflow:
```bash
# Before committing to git
./cleanup_project.sh
git status
git add .
git commit -m "Clean commit"
```

## 💡 Pro Tips

1. **Before cleanup**: Make sure you have backups of important files
2. **After cleanup**: Run `git status` to see what changed
3. **Large checkpoints**: Consider moving old checkpoints to external storage
4. **Kaggle uploads**: Clean before zipping for Kaggle to reduce upload size

## 🚨 Troubleshooting

**If cleanup script fails:**
```bash
# Make it executable
chmod +x cleanup_project.sh cleanup_project_safe.sh

# Run with bash explicitly
bash cleanup_project_safe.sh
```

**If you accidentally delete something:**
```bash
# Check git history
git status
git checkout -- <file>

# Or restore from archives
ls archives/old_files/
```

## 📈 Benefits

After cleanup:
- ✅ Faster file searches
- ✅ Cleaner git status
- ✅ Smaller project size
- ✅ Easier to navigate
- ✅ Faster Kaggle uploads
- ✅ Better IDE performance
