# SGD strategies

# using Pkg
# Pkg.add("PythonPlot")
# Pkg.add("ProgressLogging")

using Plots
using Zygote
using Distributions
using ProgressLogging


# Generate a function to minimise, like a bi-demensional gaussian region
# y = exp(-(x1^2 + x2^2))

n = 100
x1 = collect(LinRange(-3., 3., n))
x2 = collect(LinRange(-3., 3., n))

function real_f(x1, x2)
    @. 2 * exp(-(x1^2 + x2^2))
end

y = real_f(x1, x2)

Plots.contour(x1, x2, (x1, x2) -> @. 2 * exp(-(x1^2 + x2^2)))

function loss_f(theta)
    y_hat = @. 2 * exp(-((x1 - theta[1])^2 + (x2 - theta[2])^2))
    mean((y - y_hat).^2)
end

"""
    Vanilla SGD
"""
function sgd_grads_update(grads; eta)
    @. grads = eta * grads
    return grads
end

theta = [-1, 1]
Plots.scatter!(theta[1:1], theta[2:2], label="Theta")

batch_loss, batch_grads = Zygote.withgradient(theta) do params
    loss_f(params)
end

sgd_grads_update(batch_grads[1], eta=0.1)

# Ok, it seems to be working fine

# Optimisation loop
theta = [-1., 1.]

n_iter = 100
iter = 1
theta_trace = zeros(Float64, 2, n_iter)

for iter in range(1, n_iter)
    println(iter)

    # take gradients
    batch_loss, batch_grads = Zygote.withgradient(theta) do params
        loss_f(params)
    end

    # Update the gradients
    sgd_grads_update(batch_grads[1], eta=0.1)

    # Update the parameters Theta
    theta -= batch_grads[1]
    theta_trace[:, iter] = theta
    
end

Plots.contour(x1, x2, (x1, x2) -> @. 2 * exp(-(x1^2 + x2^2)))
Plots.scatter!([-1], [1], label="Theta", ms=5)

Plots.scatter!(
    theta_trace[1:1, :]',
    theta_trace[2:2, :]',
    label="Theta",
    ms=3
)

# Compare with Flux Optimiser
using Flux

# flux_sgd = Descent(0.1)
flux_sgd = RMSProp(0.1)

theta = [-1., 1.]

n_iter = 10
iter = 1
theta_trace = zeros(Float64, 2, n_iter)

for iter in range(1, n_iter)
    println(iter)

    # take gradients
    batch_loss, batch_grads = Zygote.withgradient(theta) do params
        loss_f(params)
    end

    Flux.Optimise.update!(flux_sgd, theta, batch_grads[1])

    theta_trace[:, iter] = theta
    
end

Plots.contour(x1, x2, (x1, x2) -> @. 2 * exp(-(x1^2 + x2^2)))
Plots.scatter!([-1], [1], label="Theta", ms=5)

Plots.scatter!(
    theta_trace[1:1, :]',
    theta_trace[2:2, :]',
    label="Theta",
    ms=3
)
