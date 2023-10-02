# Zygote

using Pkg
Pkg.add("Zygote")

using Zygote
using Distributions
using Random


"""
    Test with Linear Regression - MSE Loss
"""

# Generate data
n = 100
p = 4
# First coefficient is the intercept
true_beta = [0.5, 1., -1., -0.5]

X = zeros(Float32, n, p)

X[:, 1] .= 1.

# Fill 2nd and 3rd column with random Normal samples
norm_d = Distributions.Normal(1., 0.5)
X[:, 2:3] = Random.rand(norm_d, (n, 2))
# Fill 4th column with a range of floats from -1 to 1
X[:, 4] = collect(LinRange(-1., 1., n))

# Get y = X * beta + err ~ N(0, 1)
y = X * true_beta + 0.5 * Random.rand(Distributions.Normal(), n)

# Model
function lin_model(x, beta)
    x * beta
end

# Loss: MSE
function loss_mse(y_pred, y_true)
    mean((y_pred - y_true).^2)
end

# Check
loss_mse(lin_model(X, true_beta), y)

x_batch = X[1:30, :]
y_batch = y[1:30]

loss_mse(lin_model(x_batch, true_beta), y_batch)

# Analytical Gradient
XtX = 2 * x_batch'x_batch / 3
Xty = 2 * x_batch'y_batch / 3

function gradient_mse(beta)
    return XtX * beta - Xty
end

gradient_mse(true_beta)


# Using Zygote AD
batch_loss, batch_grads = Zygote.withgradient(true_beta) do params
    y_pred = lin_model(x_batch, params)
    loss_mse(y_pred, y_batch)
end

batch_loss
batch_grads

# IT WORKS !!
