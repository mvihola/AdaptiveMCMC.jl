# Further examples

Here are some further examples how the package can be used.

## Custom sampler

The package provides simple building blocks which you can use within a 'custom' MCMC sampler. Here is an example:

```julia
using AdaptiveMCMC

# Sampler in R^d
function mySampler(log_p, n, x0)

    # Initialise random walk sampler state: r.x current state, r.y proposal
    r = RWMState(x0)

    # Initialise Adaptive Metropolis state (with default parameters)
    s = AdaptiveMetropolis(x0)
    # Other adaptations are: AdaptiveScalingMetropolis,
    # AdaptiveScalingWithinAdaptiveMetropolis, and RobustAdaptiveMetropolis

    X = zeros(eltype(x0), length(x0), n) # Allocate output storage
    p_x = log_p(r.x)                     # = log_p(x0); the initial log target
    for k = 1:n

        # Draw new proposal r.x -> r.y:
        draw!(r, s)

        p_y = log_p(r.y)                      # Calculate log target at proposal
        alpha = min(one(p_x), exp(p_y - p_x)) # The Metropolis acceptance probability

        if rand() <= alpha
            p_x = p_y

            # This 'accepts', or interchanges r.x <-> r.y:
            # (NB: do not do r.x = r.y; these are (pointers to) vectors!)
            accept!(r)
        end

        # Do the adaptation update:
        adapt!(s, r, alpha, k)

        X[:,k] = r.x   # Save the current sample
     end
    X
end

# Standard normal target for testing
normal_log_p(x) = -mapreduce(e->e*e, +, x)/2

# Run 1M iterations of the sampler targetting 30d standard Normal:
X = mySampler(normal_log_p, 1_000_000, zeros(30))
```

## Using custom 'local move' kernel within adaptive parallel tempering

