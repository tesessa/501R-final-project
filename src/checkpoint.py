import torch

checkpoint = torch.load(
    "/home/tessa343/classes/501R-final-project/src/va_model/pytorch_model.bin",
    map_location='cpu',
    weights_only=False
)

print("Checkpoint type:", type(checkpoint))
print("\nAll keys in checkpoint:")
for key in sorted(checkpoint.keys()):
    print(f"  {key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else type(checkpoint[key])}")