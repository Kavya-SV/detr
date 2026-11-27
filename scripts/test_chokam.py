import torch
from ultralytics import RTDETR

def test_chokam_integration():
    print("Initializing RT-DETR model...")
    # Load the base model structure, this will initialize our updated head.py and chokam.py
    model = RTDETR("rtdetr-l.yaml")
    
    # We need to access the underlying PyTorch nn.Module which is model.model
    pytorch_model = model.model
    pytorch_model.train() # Set to train mode
    pytorch_model.nc = 80 # required for loss computation
    
    device = next(pytorch_model.parameters()).device
    
    B, C, H, W = 2, 3, 640, 640
    print(f"Creating dummy batch of images shape {(B, C, H, W)}...")
    img = torch.randn(B, C, H, W, device=device)
    
    # Dummy targets
    # batch_idx, cls, cx, cy, w, h
    # We will just create 2 dummy bboxes per image
    bboxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.3, 0.3, 0.1, 0.1],
        [0.7, 0.7, 0.3, 0.3],
        [0.8, 0.2, 0.1, 0.4]
    ], device=device)
    batch_idx = torch.tensor([0, 0, 1, 1], device=device)
    cls_idx = torch.tensor([0, 1, 0, 2], device=device)
    
    targets = {
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "cls": cls_idx,
        "img": img
    }
    
    print("Running forward pass and computing loss...")
    loss, loss_items = pytorch_model.loss(batch=targets, preds=pytorch_model(img))
    
    print("\n--- Chokam Routing Statistics ---")
    head = pytorch_model.model[-1]
    
    total = head.stat_total
    high = head.stat_high
    med = head.stat_med
    low = head.stat_low
    recovered = head.stat_recovered
    
    print(f"Total Tokens: {total}")
    print(f"High: {high / total * 100:.2f}%")
    print(f"Medium: {med / total * 100:.2f}%")
    print(f"Low: {low / total * 100:.2f}%")
    print(f"After refinement:")
    print(f"Recovered: {recovered / total * 100:.2f}%")
    print(f"Finally kept: {(high + recovered) / total * 100:.2f}%")
    print(f"Finally pruned: {total - (high + recovered)} / {total} ({(total - (high + recovered)) / total * 100:.2f}%)")
    
    print("\n--- Gradient Check ---")
    print("Running backward pass...")
    loss.backward()
    
    # Check if Chokam Scorer got gradients
    scorer_grad = False
    for name, param in head.chokam_scorer.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            scorer_grad = True
            break
            
    # Check if Refiner got gradients
    refiner_grad = False
    for name, param in head.chokam_refiner.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            refiner_grad = True
            break
            
    print(f"Chokam Token Scorer received gradients: {scorer_grad}")
    print(f"Chokam Refiner received gradients: {refiner_grad}")
    
    if scorer_grad and refiner_grad:
        print("\nSUCCESS! The hard pruning architecture is fully differentiable and working.")
    else:
        print("\nWARNING: Gradients did not flow correctly.")

if __name__ == "__main__":
    test_chokam_integration()
