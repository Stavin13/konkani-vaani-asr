# Quick Fix for Cell 2 Error

## The Problem:
Cell 2 didn't find your files, so Cell 3 failed.

## The Solution:

### **Step 1: Add the Shared Folder**

1. Open this link in a new tab:
   ```
   https://drive.google.com/drive/folders/1KX7k_z2negFKq3qFjHJh-K1U-MEcNp7P
   ```

2. You'll see a folder with 3 files

3. At the top, click the **folder name**

4. Click **"Add shortcut to Drive"** (⭐ star icon)

5. Choose **"My Drive"**

6. Click **"Add shortcut"**

### **Step 2: Re-run Cell 2**

Go back to Colab and click "Run" on Cell 2 again.

It should now show:
```
✅ checkpoint_epoch_15.pt (293.9 MB)
✅ konkani_project.zip (14600.0 MB)  
✅ vocab.json (0.0 MB)
```

### **Step 3: Continue with Cell 3**

Once Cell 2 shows all ✅ marks, run Cell 3.

---

## Alternative: Manual Path (if automatic search fails)

If Cell 2 still can't find files, run this in a new cell:

```python
# Manual search
!find /content/drive -name "checkpoint_epoch_15.pt" 2>/dev/null
```

You'll see something like:
```
/content/drive/MyDrive/some_folder_name/checkpoint_epoch_15.pt
```

Copy the folder path (everything before `/checkpoint_epoch_15.pt`) and run:

```python
# Replace with your actual path
folder_path = "/content/drive/MyDrive/some_folder_name"

# Save it
with open('/tmp/folder_path.txt', 'w') as f:
    f.write(folder_path)

print(f"✅ Saved path: {folder_path}")
```

Then continue with Cell 3.

---

## Why This Happens:

The shared folder link gives you access to the files, but you need to add a "shortcut" to your Drive so Colab can find them easily. Think of it like creating a bookmark.

Once you add the shortcut, Cell 2 will work!
