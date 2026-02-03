from typing import Callable

import torch
from torch import Tensor, nn

from .integrator import Integrator


def gradient(y: Tensor, x: Tensor) -> Tensor:
    """Calculate the gradient of y with respec to x
    Wrapper function for torch.autograd.grad for ease of use

    Initiates grad_outputs as ones, and doesn't create the graph by default

    Args:
        y (Tensor): the output of a function to take grad from
        x (Tensor): the input to the function to take grad w.r.t

    Returns:
        Tensor: grad of y w.r.t x
    """

    grad_outputs = torch.ones_like(y).detach()

    grad = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=False)[0]

    return grad


class ODEProcess:
    """
    Process solver of the vector field that can sample the probability path arbitrarily
    or compute the likelihood along the path
    """

    def __init__(self, vector_field: nn.Module, integrator: Integrator) -> None:
        self.vector_field = vector_field
        self.integrator = integrator

    def sample(
        self, x_init: Tensor, ints: Tensor, steps: int, **vf_extras
    ) -> tuple[Tensor, Tensor]:
        """Integrates the vector field along the probability path within the time provided
        Sampling can run in reverse if time is in descending order, but x_init must also match

        Args:
            x_init (Tensor): x(t[0]) the initial condition of the ODE, size (B, D...)
            ints (Tensor): start and end point of the interval in which to integrate, size (B, 2)
            steps (int): number of steps for the ODE solver
            **vf_extras: additional parameters for the model if needed

        Returns:
            tuple[Tensor, Tensor]: (steps+1, B, 1) of time
            and (steps+1, B, D...) of solution trajectories
        """

        # function to integrate over time
        def diff_eq(t: Tensor, x: list[Tensor]) -> list[Tensor]:
            _x = x[0]
            return [self.vector_field(_x, t, **vf_extras)]

        with torch.no_grad():
            t_traj, x_traj = self.integrator.integrate(
                diff_eq, [x_init], ints, steps=steps
            )

        return t_traj, x_traj[0]

    def compute_likelihood_once(
        self,
        x1: Tensor,
        ints: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        steps: int,
        **vf_extras
    ) -> tuple[Tensor, Tensor]:
        """Calculates the unbiased estimation of log p1 using Hutchinson's estimator

        Args:
            x1 (Tensor): target samples from t=1 for which to calculate log p1
            t (Tensor): time points in which to integrate in descending interval [1, 0], size (N)
            log_p0 (Callable[[Tensor], Tensor]): function that calculates log likelihood
                at t=0 given points at t=0
            steps (int): number of steps for the ODE solver

        Returns:
            tuple[Tensor, Tensor]: a (B, D) tensor of the positions at t=0
            and a (B,1) tensor of likelihoods for those positions
        """

        # make Z for Hutchinson's, it needs E[Z] = 0 and Cov(Z, Z) = 1
        # Rademacher distribution sampling
        z = (torch.randn_like(x1) < 0) * 2.0 - 1.0

        # ut function to integrate over time
        def diff_eq(t: Tensor, x: Tensor) -> Tensor:
            return self.vector_field(x, t, **vf_extras)

        # the dynamics function of the system
        # two to find are the positions at t and the divergence at t
        # the positions will then be used to calculate log p0
        # and the divergence will be added to it
        def dynamics_eq(t: Tensor, states: list[Tensor]) -> list[Tensor]:
            # don't need div at t, since we won't be using it, just the xt
            xt, _ = states

            with torch.set_grad_enabled(True):
                xt.requires_grad_()  # needed xt grads for Hutchinson's
                ut = diff_eq(t, xt)

                # Hutchinson's divergence estimator E[Z^T D_x(ut) Z]
                # D_x(ut) the jacobian of ut at x

                # calculate ut @ z, then take grad, since z is constant w.r.t x
                # this will make d_x ut @ z, so no need to grad first saves 1 op
                ut_dot_z = torch.einsum(
                    "nm,nm->n", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_ut_dot_z = gradient(ut_dot_z, xt)

                div = torch.einsum(
                    "nm,nm->n",
                    grad_ut_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

            return [ut.detach(), div.unsqueeze(1).detach()]

        init_states = [x1, torch.zeros((x1.shape[0], 1), device=x1.device)]

        with torch.no_grad():
            _, (sol, div) = self.integrator.integrate(
                dynamics_eq, init_states, ints, steps=steps
            )

        x0 = sol[-1]
        log_p_x0 = log_p0(x0)

        div_x0 = div[-1].squeeze(1)

        return x0, log_p_x0 + div_x0

    def compute_likelihood(
        self,
        x1: Tensor,
        ints: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        steps: int,
        est_steps: int = 10,
        **vf_extras
    ) -> tuple[Tensor, Tensor]:
        """Calculates the unbiased estimation of log p1 using Hutchinson's estimator,
        Estimates `est_steps` times and then aggregates by averaging

        Args:
            x1 (Tensor): target samples from t=1 for which to calculate log p1
            t (Tensor): time points in which to integrate in descending interval [1, 0], size (N)
            log_p0 (Callable[[Tensor], Tensor]): function that calculates log likelihood
                at t=0 given points at t=0
            steps (int): number of steps for the ODE solver
            est_steps (int, optional): number of estimations before aggregation. Defaults to 10.

        Returns:
            tuple[Tensor, Tensor]: a (B, D) tensor of the positions at t=0
            and a (B,1) tensor of likelihoods for those positions
        """

        log_p = 0.0
        sol = 0.0

        for _ in range(est_steps):
            sol_est, log_p_est = self.compute_likelihood_once(
                x1, ints, log_p0, steps, **vf_extras
            )

            log_p += log_p_est
            sol += sol_est

        sol = sol / est_steps
        log_p = log_p / est_steps

        return sol, log_p  # type: ignore


class PotentialProcess:
    """
    Process solver for the gradient based potential manifold,
    that can sample points in the field using the potential
    """

    def __init__(self, potential_manifold: nn.Module, integrator: Integrator) -> None:
        self.potential_manifold = potential_manifold
        self.integrator = integrator

    def sample(
        self, x_init: Tensor, ints: Tensor, steps: int, **pm_extras
    ) -> tuple[Tensor, Tensor]:
        """Integrates the point in vector space using the potential field along the probability path

        Args:
            x_init (Tensor): initial condition of the Process
            ints (Tensor): start and end time points for each point in x_init
            steps (int): number of steps for the Process Integrator

        Returns:
            tuple[Tensor, Tensor]: (steps+1, B, 1) of time
            and (steps+1, B, D...) of solution trajectories
        """

        def diff_eq(t: Tensor, x: list[Tensor]) -> list[Tensor]:
            _x = x[0]

            # to get the velocity we need to allow _x to have grads
            _x.requires_grad_(True)

            with torch.set_grad_enabled(True):
                potential: Tensor = self.potential_manifold.forward(_x, t, **pm_extras)
                velocity = -gradient(potential.sum(), _x)

            return [velocity]

        with torch.no_grad():
            t_traj, x_traj = self.integrator.integrate(
                diff_eq, [x_init], ints, steps=steps
            )

        return t_traj, x_traj[0]
