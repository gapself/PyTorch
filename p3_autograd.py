import torch

x = torch.rand(3, requires_grad=True)
print(x)
y = x + 2
print(y)

print('\n\nEXAMPLE 1 - CALCULATE GRADIENTS with SCALAR:')
z = y * y * 2
print(z)
z = z.mean()  # tensor has only one value
print(z)
z.backward()  # method which generates gradient of z with respect to x == dz/dx:
# in .backward() method here we dont need to use argument inside, cause z is scala value
# grad can be produced only for scalar outputs, if there is no scalar value, we give it a vector -- EXAMPLE 2
print('here are stored gradients:')
print(x.grad)
print(
    '* GRAD - it also produces Vector Jacobian products in background - chain rule - later it is J*v, so we need to multiply (J * vector) \n')

print('\n\nEXAMPLE 2 - CALCULATE GRADIENTS without SCALAR')
w = y * y * 2
print(w)
# w.backward() #RuntimeError: grad can be implicitly created only for scalar outputs!!
# we need to give gradient arguments, so we create vector of the same size as x
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
w.backward(v)
print(x.grad)

print('\n\nEXAMPLE 3 - HOW TO NOT TRACK GRADIENTS, Y without gradient function')
x = torch.rand(3, requires_grad=True)
print(x)
# first option - x.requires_grad_(False)
x.requires_grad_(False)
print(x)
# second option - x.detach() -> this will create new tensor that doesnt require gradient
y = x.detach()
print(y)
# third option - with torch.no_grad()
with torch.no_grad():
    y = x + 2
    print(y)

print(
    '\n\nEXAMPLE 4 - WHENEVER WE CALL THE BACKGROUND FUNCTION THEN THE GRADIENT FOR THIS TENSOR WILL BE ACCUMULATED IN THE GRAD ATTRIBUTE, VALUES WILL BE SUMMED UP ')
print('TRAINING EXAMPLE with some weights: ')
weights = torch.ones(4, requires_grad=True)
for epoch in range(
        1):  # make at first only one iteration, and then check what will happen if we make 2 or more iterations
    model_output = (weights * 3).sum()  # simulate whatever
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()  # ! empty the gradient before next iteration (second or <)

optimizer = torch.optim.SGD(weights, lr=0.01)  # stochastic gradient descent
optimizer.step()
optimizer.zero_grad()
