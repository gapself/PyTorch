import torch
import numpy as np

x = torch.empty(3)  # 1D vector with 3 elements
print(x)

x = torch.empty(2, 3)  # 2D vector 6 el, 3 in each col
print(x)

x = torch.empty(2, 2, 3)  # 3D vector 12 el
print(x)

x = torch.rand(2, 2)
print(x)

x = torch.zeros(2, 2)
print(x)

x = torch.ones(2, 2)
print(x)
print(x.dtype)

x = torch.ones(2, 2, dtype=torch.int)
print(x.dtype)

x = torch.ones(2, 2, dtype=torch.float16)
print(x.dtype)

print(x.size())

x = torch.tensor([2.5, 0.1])
print(x)
print(x.size())

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)

z = x + y
print('\n\nELEMENT WISE ADDITION:')
print(z)
z = torch.add(x, y)
print('\n\nMETHOD ADDITION:')
print(z)

y.add_(x)
print('\n\nIN PLACE ADDITION (keep in mind that it modifies our variable in which it is applied!):')
print(y)

z = x - y
z = torch.sub(x, y)
print(z)

z = x * y
z = torch.mul(x, y)
y.mul_(x)
print(z)
print(y)

z = x / y
z = torch.div(x, y)
print(z)

print('\n\n\nSLICE OPERATOR')
x = torch.rand(5, 3)
print(x)
print(x[:, 0])  # we want all rows but 1 col
print(x[0, :])  # 1nd row and all cols
print(x[1, :])  # 2nd row and all cols
print(x[1, 1])  # value from 2nd row 2nd col
print(x[1, 1].item())  # takes value from tensor if its one element !!

print('\n\n\nVIEW method returns tensor with the same data as the self tensor but of a different shape')
x = torch.rand(4, 4)
print(x)
# num of elements must be still the same after initialized VIEW
y = x.view(16)  # 1D VECTOR
print(y)

y = x.view(-1, 8)  # resize to 2D 2 rows 8 cols
print(y)
print(y.size())

print('\n\n\n\n!!! WARNING GPU CPU - if a tensor is on CPU not GPU both objects will share same memory location, changing one\'ll change the other')
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))
print(b)
a.add_(1)  # so if we add 1 to a
print(a)
print(b)  # it is added also to b

print('\n\n\nTENSOR from NUMPY')
c = np.ones(5)
print(c)
d = torch.from_numpy(c)
print(d)

c += 1
print(c)
print(d)  # also modified

# torch CREATED on GPU if you have cuda toolkit
# on mac it'll return false (?)
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)  # to GPU
    z = x + y
    # z.numpy() # cannot convert GPU tensor back to NUMPY (numpy handle CPU tensor)
    z = z.to('cpu')  # back to CPU
# IT IS NOT AVAILABLE so nth'll happen

print('\n\n\nrequires_grad - tells pytorch to calculate later gradient steps, if we want to optimize our model we need to specify this keyword')
f = torch.ones(5, requires_grad=True)
print(f)
