#!/usr/bin/env python3
try:
    import transformers
    print("Transformers imported successfully!")
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Error importing transformers: {e}")
    import traceback
    traceback.print_exc()