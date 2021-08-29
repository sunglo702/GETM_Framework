import torch
import numpy as np


# initialize from directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# initialze from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# initialize from another tensor
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(f'Ones Tensor is \n{x_ones}\n')
print(f'Rand Tensor is \n{x_rand}\n')

# initialize from a shape
shape = (2,3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)

print(f'rand tensor is {rand_tensor}')
print(f'ones tensor is {ones_tensor}')
print(f'zero tensor is {zero_tensor}')

# Attribute of a tensor
tensor = torch.rand(3,4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')

# if gpu is posiible
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# slicing and standar numpy-like indexing
tensor = torch.ones(4,4)
print(f'First row: ', tensor[0])
print(f'First column: ', tensor[:,0])
tensor[..., -2] = 2
print(f'Last row: ', tensor[:, -1])
print(tensor)

# joining tensor
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

tensor = torch.ones(3,3)
# arithmetic operations
y1 = tensor @ tensor.T
print (y1)
y2 = tensor.matmul(tensor.T)
print (y2)
y3 = torch.rand_like(tensor)
print(f'y3 is {y3}')
print(torch.matmul(tensor, tensor.T, out=y3))

# compute the element-wise product
z1 =tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print(torch.matmul(tensor, tensor.T, out=z3))

# single element tensors
agg = tensor.sum()
agg_item = agg.item()
print(f'{agg}, {agg_item}')
print(agg_item, type(agg_item))

# In-place operations Operations that store the result into the operand are called in-place.
# They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
print(tensor)
tensor[:,-2] = 3
tensor.add_(5)
print(tensor)

# bridge with numpy
t = torch.ones(5,5)
print(f'{t}')
n = t.numpy()
print(f'{n}')
t = t.add_(3)
print(f'{t}')
print(f'{n}')

n = np.ones(5)
t = torch.from_numpy(n)
print(f'{t}')
print(f'{n}')
np.add(n, 1,  out=n)
print(f'{t}')
print(f'{n}')



