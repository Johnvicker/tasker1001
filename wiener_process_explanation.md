# The Wiener Process in Financial Mathematics

A **Wiener Process** (often denoted as $W_t$), also known as standard Brownian motion, is a continuous-time stochastic process that serves as the foundation for modeling random asset price movements in calculus-based finance (like the Black-Scholes model).

## Key Mathematical Properties

For a process $W_t$ to be a Wiener process, it must satisfy:

1.  **$W_0 = 0$**: The process starts at zero.
2.  **Independent Increments**: The changes in the process over non-overlapping time intervals are independent. For $0 \le s < t < u < v$, the increment $W_t - W_s$ is independent of $W_v - W_u$. Memory of the past does not influence the future step (Markov property).
3.  **Gaussian Increments**: The increment $W_t - W_s$ follows a normal distribution with mean 0 and variance $t-s$:
    $$W_t - W_s \sim \mathcal{N}(0, t-s)$$
4.  **Continuous Paths**: The function $t \mapsto W_t$ is continuous almost everywhere.

## Role in Geometric Brownian Motion (GBM)

In our model, we use GBM to simulate the price $S_t$:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Here, $dW_t$ represents the "random shock" or noise driving the price at each infinitesimal time step.
- $\sigma S_t dW_t$ is the volatility term.
- Since $dW_t \sim \mathcal{N}(0, dt)$, the randomness scales with the square root of time ($\sqrt{dt}$), which is why volatility is often referred to as the "square root of time" rule.

In the discrete simulation (Monte Carlo), we approximate $dW_t$ as:
$$dW_t \approx \sqrt{\Delta t} \cdot Z$$
where $Z$ is a standard normal random variable ($Z \sim \mathcal{N}(0, 1)$).
