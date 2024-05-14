import torch

# Create a 3D tensor
tensor_3d = torch.rand(2, 3, 4)
print("Original tensor shape:", tensor_3d.shape)


# Convert torch.Size to list
shape_list = list(tensor_3d.shape)
print("Shape as list:", shape_list)

# Convert list back to torch.Size
shape_torch = torch.Size(shape_list)
print("Shape as torch.Size:", shape_torch)


# Flatten the tensor
tensor_1d = tensor_3d.view(-1)
print("Flattened tensor shape:", tensor_1d.shape)

# Recover the original shape
tensor_recovered = tensor_1d.view(shape_torch)
print("Recovered tensor shape:", tensor_recovered.shape)