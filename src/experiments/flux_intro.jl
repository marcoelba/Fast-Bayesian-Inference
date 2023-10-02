# Flux

# using Pkg
# Pkg.add("Flux")

using Flux
using Plots

# Example of Linear Regression estimated using Flux
x = hcat(collect(Float32, -3:0.1:3)...)
n = length(x)

# @. to broadcast the operation with a scalar additive constant
function model(x)
    @. 3 * x + 2
end

y = model(x) + rand(Float32, (1, n))

# PLOT
Plots.plot(
    vec(x), vec(y),
    lw=3,
    seriestype=:scatter,
    label="",
    title="Generated data",
    xlabel="x",
    ylabel= "y"
);

# Model with ML notation
function ml_model(W, b, x)
    @. W * x + b
end

# Initialise the parameters
W = rand(Float32, 1, 1)
b = [0.0f0]

ml_model(W, b, x) |> size

function custom_loss(W, b, x, y)
    y_hat = ml_model(W, b, x)
    sum((y .- y_hat).^2) / length(x)
end

custom_loss(W, b, x, y)

# Define Flux model using Flux layers
flux_model = Flux.Dense(1 => 1, identity, bias=true)
(flux_model.weight, flux_model.bias)

flux_model(x) |> size

# Loss function in Flux notation
function flux_loss(flux_model, x, y)
    y_hat = flux_model(x)
    Flux.mse(y_hat, y)
end

flux_loss(flux_model, x, y)


# Take the gradient using Zygote, used via Flux
dLdW, dLdb, _, _ = Flux.gradient(custom_loss, W, b, x, y);


"""
    Train the Fulx model with an explicit training loop
"""

# Loss to use during training
function my_loss(y_pred, y_true)
    Flux.mse(y_pred, y_true)
end

# Zip X and y into a single object for batches
input_data = zip(
    eachslice(x, dims=2),
    eachslice(y, dims=2)
)

# Setup the Optimiser
opt_state = Flux.setup(Descent(), flux_model)

train_log = []

for epoch in 1:10
    losses = Float32[]

    # loop over batches
    iter = 0
    for (iter, data) in enumerate(input_data)
        features, label = data

        batch_loss, grads = Flux.withgradient(flux_model) do m
            # Anything inside this block is differentiated
            model_preds = m(features)
            my_loss(model_preds, label)
        end

        # save the loss
        push!(losses, batch_loss)

        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(batch_loss)
            @warn "loss is $batch_loss on item $iter" epoch
            continue
        end

        Flux.update!(opt_state, flux_model, grads[1])
    end

    push!(train_log, (; losses))

end

train_log[1][1]
Flux.params(flux_model)
opt_state


# Code Definition of Julia Optimiser
abstract type AbstractOptimiser end

mutable struct MyDescent <: AbstractOptimiser
    eta::Float64
end

MyDescent() = MyDescent(0.1)

# Add the new method to the multiple dispatch of "apply!"
function apply!(o::MyDescent, params, params_grads)
    params_grads .*= o.eta
end

my_opt = MyDescent(10)
my_params = (a=[1.], b=[0.5])
my_grads = (a=0.1, b=10.)

apply!(my_opt, my_params, my_grads)

new_opt = Descent(10.)

Flux.Optimisers.update!(new_opt, my_params, my_grads)


