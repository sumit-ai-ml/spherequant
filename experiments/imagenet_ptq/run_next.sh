#!/bin/bash
# Chain: BN recal -> ConvNeXt -> EfficientNet
cd /home/sumit-pandey/Documents/turboquant/turboquant-rs/experiments/imagenet_ptq
echo "=== BN recal sweep starting $(date) ==="
python -u run_bn_recal.py --models resnet18 resnet50 mobilenet_v2 --bits 4 6 --codebooks beta > results/bn_recal.log 2>&1
echo "=== BN recal done $(date), starting ConvNeXt ==="
python -u run.py --models convnext_tiny --bits 2 4 6 8 --codebooks uniform beta --batch-size 32 > results/convnext_main.log 2>&1
python -u run_quarot.py --models convnext_tiny --bits 2 4 6 8 --batch-size 32 > results/convnext_quarot.log 2>&1
echo "=== ConvNeXt done $(date), starting EfficientNet ==="
python -u run.py --models efficientnet_b0 --bits 2 4 6 8 --codebooks uniform beta --batch-size 64 > results/effnet_main.log 2>&1
python -u run_quarot.py --models efficientnet_b0 --bits 2 4 6 8 --batch-size 64 > results/effnet_quarot.log 2>&1
echo "=== All done $(date) ==="
