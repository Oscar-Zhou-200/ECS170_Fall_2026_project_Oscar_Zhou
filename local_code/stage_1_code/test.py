import torch
import sklearn

print("PyTorch version:", torch.__version__)
print("scikit-learn version:", sklearn.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)