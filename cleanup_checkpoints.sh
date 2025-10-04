#!/bin/bash

echo "🧹 Cleaning up unnecessary checkpoint files for deployment..."
echo "Keeping only best.pth and deleting intermediate checkpoints..."

cd checkpoints

# Count files before cleanup
BEFORE=$(ls -1 checkpoint_epoch_*.pth 2>/dev/null | wc -l)

# Remove all intermediate checkpoints, keep only best.pth
rm -f checkpoint_epoch_*.pth

# Count files after cleanup
AFTER=$(ls -1 *.pth 2>/dev/null | wc -l)

echo "✅ Cleanup complete!"
echo "   Removed: $BEFORE checkpoint files"
echo "   Remaining: $AFTER file(s) (best.pth)"

cd ..

# Show new size
echo ""
echo "📦 New checkpoints folder size:"
du -sh checkpoints/
