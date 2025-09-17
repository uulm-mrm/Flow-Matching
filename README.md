# Flow Matching
Backend for creating Flow Matching (FM) models. With Schedulers, Paths and Integrators available to make creating new Flow Matching models easier.

Flow matching models are a type of generative model that models densities. At its core it is a physics based model that transporst points from `t=0` to `t=1`, points at `t=0` are sampled from some known distribution, while points at `t=1` are sampled from the target distribution that the model learns.

The points are moved from `t=0` to `t=1` along paths, since the origin and the enpoint of this path is a distribution, at each time `t` along the path there is another distribution, hence the path is called the `probability path`, but other than specific cases along this probability path, the probabilities aren't tangible, unless a nontrivial integral is solved.

The actual neural network part of the model is the vector field that pushes points from `t=0` to `t=1` along the path at each timestep.

## Theory

This section is for explaining the building blocks of the FM model. All of this is basically made from by following the `Flow Matching Guide and Code` book/paper from Meta.

### Probability paths

A probability path is the name of the path a point takes while travelling from `t=0` to `t=1`. This path is ideally affine (and this is how it is implemented), because straight simple paths are easy to integrate with low error. The affine path is defined by a function that decides where a point should be, this function is defined as:

$$ x_t = \alpha(t)  x_1 + \sigma(t)  x_0 $$

Where $x_1$ is the point at `t=1`, usually sampled from the actual dataset, while $x_0$ is the point at `t=0` which is usually sampled from Gaussian noise $\mathcal{N}(\bf{0}, I)$, or any other arbitrary noise function. Gaussians are mostly used since they are affine invariant, and since we're using affine paths, this guy is the best.

Since we're learning vector fields, we're also interested in velocities along these probability paths. And since velocity is just the $\frac{d}{dt}x$ we need to find the first derivative over time of the above function for calculating positions along the path.

$$ \dot{x_t} = \alpha'(t)  x_1 + \sigma'(t)  x_0 $$

### Schedulers

The $\alpha$ and $\sigma$ functions are governed by schedulers. The schedulers implemented are convex, meaning that they gradually go from being weighted more towards $x_0$ and as time goes on they tend towards $x_1$. But in theory they can also be more or less anything, just as long as they are somewhat stable (looking at you variance exploding and variance perserving schedulers)

The simplest scheduler comes from the optimal transport theorem, and is therefore named `OTScheduler`, in which $\alpha(t) = t$ and $\sigma(t) = 1-t$. Another scheduler that is simple and common is the `Polynomial Scheduler` which is just the `OT` one except, `t` is exponentiated by a parameter, meaning: $\alpha(t) = t^n$ and $\sigma(t) = 1 - t^n$.

### Velocity Fields

The actual neural network part of the FM model is the velocity vector field. It learns to assign each point at some time a velocity along all of its dimensions. Therefore the input to this type of network is some $n$ dimensional tensor, while the output must also be $n$ dimensional as each component of the input needs a speed assigned to it.

There are two constraints the model has to meet, one is simple enough, and required for sampling and likelihood calculation later on, the other is a bit more troubling.

The first constraint is that the overall network must in the end be differentiable w.r.t the input which is more or less any neural network.

The other constraint is that the network needs to be aware of time as a concept in some way. Usually the time dimension is added as another dimension to the input vector, but this is not always true (i.e. time dependent CNNs). So keep this in mind when making networks.

Keep in mind, **Network outputs are just speeds not the actual points in space. To get actual points you need to integrate!**

### Loss

The loss is super simple, the network needs to learn a vector field, by assigning a speed to all the components of the input tensor.

$$ \mathcal{L} = \frac{1}{n}\Sigma\left(u^\theta_t - \dot{x}_t\right)^2 $$

Which is basically just `MSE` on the predicted speed $u^\theta_t$ and what the path predicts the speed should be $\dot{x}_t$.

This is a super simple convex loss objective, so the network converges nicely. Don't be too scared when losses aren't going down for a long time, this loss tells the network to almost exactly match the speeds which is near impossible for too many points and dimensions. But a nice thing is that the good ol' rule applies here, more iterations means better representation.

### Sampling

Since FM is a generative model at its core, sampling points is obviously very important. Like we said in the velocity fields section, the network produces speeds, so the integral of the speed from `t=0` to `t=1` is the actual position of the point in space. An intuitive way to look at the integral is through Verlet integration (which is in reality just Euler's method for function approximation):

$$ x_{t+dt} = x_t + u_t(x_t)dt $$

So for small $dt$ this converges nicely. Obviously Euler isn't the only method, and one can pick from many, personally the `midpoint` method is really good.

The framework has an `Integrator` class that takes as a parameter the network that produces the velocity field. The integrator solves the integral with given parameters and produces points sampled along the path at anchors given through the `t` parameter.

### Likelihood Computation

But FM doesn't only generate new data, it can also calculate the likelihoods along the path, but in reality the only time the likelihood can be calculated at is at `t=0` since we know the exact likelihood function only at that time. So the likelihood computation consists of taking a point at `t=1` integrating in reverse from `t=1` to `t=0`, taking the produced point's position at `t=0` and calculating the likelihood for it there.

In the backend, the likelihood is governed by a dynamics function that follows both the divergence and the position of the point in time. The divergence is super important because of the `Mass Conservation Law` (MCL) from physics which states that the change of mass should always be $0$, so there is a function called the divergence which forces the change of mass to always be $0$, so that no mass is destroyed or created over time.

$$ \frac{d}{dt}m + div = 0 $$

In our context the `Mass` part of MCL is changed with probability, and now it's called the `Continuity Law`, which says probability is neither created nor destroyed over time.

$$ \frac{d}{dt}p_t({\psi_t(x_t)}) + div(u_t)(\psi_t(x_t)) = 0 $$

$\psi_t(x_t)$ here is stolen from flow networks, it just means push-forward $x$, or in other words, the position of $x$ at time $t$.

Since $div(f)(x) = Tr\left[\partial f_x(x)\right]$, or in english, the trace of the Jacobian of $f$ at $x$, is really hard to calculate, an unbiased Hutchinson's estimator is used to compute it, not giving exact divergence but near perfect. You can also run it a few times and average the results to get more precise divergences.