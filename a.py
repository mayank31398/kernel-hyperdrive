import torch


# Step 1: Create a leaf tensor
x = torch.randn(3, 3, requires_grad=True)

# Step 2: Perform operations to get a non-leaf tensor
y = x + 1  # This is a non-leaf tensor
z = y.sum()  # This is another non-leaf tensor

print(x.is_leaf)  # Output: True
print(y.is_leaf)  # Output: False
print(z.is_leaf)  # Output: False
