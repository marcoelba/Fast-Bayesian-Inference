# Bayesian Models

using Distributions
using LinearAlgebra
using Random


# Generate data
n = 1000
p = 4
# First coefficient is the intercept
true_beta = [0.5, 1., -1., -0.5]

X = zeros(Float64, n, p)

X[:, 1] .= 1.

# Fill 2nd and 3rd column with random Normal samples
norm_d = Distributions.Normal(1., 0.5)
X[:, 2:3] = Random.rand(norm_d, (n, 2))
# Fill 4th column with a range of floats from -1 to 1
X[:, 4] = collect(LinRange(-1., 1., n))

# Get y = X * beta + err ~ N(0, 1)
y = X * true_beta + 0.5 * Random.rand(Distributions.Normal(), n)


"""
    Run Metropolis-Hastings - Known y variance
"""
# Chain params
n_iter = 1000

# Prior Distributions:
# beta_0 ~ N(mu=0, sigma=10)
# sigma = 0.5

# beta container
beta_hat_trace = zeros(Float64, p, n_iter)
# intial value
beta_hat_trace[:, 1] = [0., 0.1, -0.1, 0.1]
beta_hat_t = beta_hat_trace[:, 1]

# Prior distribution (Isotonic Multivariate Normal = Indipendent components (specify the SD))
prior_d = Distributions.MvNormal(zeros(p), 10)

# Likelihood: y ~ N(X * beta_t, sigma=0.5)
# loglikelihood
function loss(;beta, X=X, y=y)
    y_mu = X * beta
    y_d = Distributions.MvNormal(y_mu, 0.5)

    return Distributions.loglikelihood(y_d, y) + Distributions.loglikelihood(prior_d, beta)
end

loss_beta_t = loss(beta=beta_hat_t)
loss(beta=true_beta)

# Proposal distribution eta: Normal(eta_mu=0, eta_sigma)
eta_d = Distributions.MvNormal(zeros(p), 1)
eta_sigma = 0.05

iter = 1
acc_count = 0

for iter in range(2, n_iter)
    println(iter)
    # propose new value for beta
    beta_star = eta_sigma * rand(eta_d) + beta_hat_t

    loss_beta_star = loss(beta=beta_star)
    ratio = loss_beta_star - loss_beta_t

    # Acceptance step
    u = rand()
    if log(u) <= ratio
        acc_count += 1
        beta_hat_t = beta_star
        loss_beta_t = loss_beta_star
    end
    beta_hat_trace[:, iter] = beta_hat_t

end

acc_count

beta_hat_trace[:, 500:n_iter]


using Plots

Plots.plot(LinearAlgebra.transpose(beta_hat_trace))

