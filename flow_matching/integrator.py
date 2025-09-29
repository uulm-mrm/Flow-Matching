from typing import Callable, Optional
from torchdiffeq import odeint

import torch
from torch import Tensor, nn


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


class Integrator:
    """
    Integrator of the vector field that can sample the probability path arbitrarily
    or compute the likelihood along the path
    """

    def __init__(self, vector_field: nn.Module) -> None:
        self.vector_field = vector_field

    def sample(
        self,
        x_init: Tensor,
        t: Tensor,
        method: str = "midpoint",
        step_size: Optional[float] = None,
        **vf_extras
    ) -> Tensor:
        """Integrates the vector field along the probability path within the time provided
        Sampling can run in reverse if time is in descending order, but x_init must also match

        Args:
            x_init (Tensor): x(t[0]) the initial condition of the ODE
            t (Tensor): time points in which to integrate, size (N)
            method (str, optional): method which to use for integration.
                Check torchdiffeq odeint for all possibilities.
                Defaults to "midpoint".
            step_size (Optional[float], optional): if the method allows for a step size,
                the step size to use. Check torchdiffeq odein if possible valid
                Defaults to None.
            **vf_extras: additional parameters for the model if needed

        Returns:
            Tensor: an (N, B, D) tensor of N solutions anchored to specific times in t
        """

        # function to integrate over time
        def diff_eq(t: Tensor, x: Tensor) -> Tensor:
            return self.vector_field(x, t, **vf_extras)

        ode_options = {"step_size": step_size} if step_size else {}

        with torch.no_grad():
            sols = odeint(diff_eq, x_init, t, method=method, options=ode_options)

        return sols  # type: ignore

    def compute_likelihood_once(
        self,
        x1: Tensor,
        t: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        method: str = "midpoint",
        step_size: Optional[float] = None,
        **vf_extras
    ) -> tuple[Tensor, Tensor]:
        """Calculates the unbiased estimation of log p1 using Hutchinson's estimator

        Args:
            x1 (Tensor): target samples from t=1 for which to calculate log p1
            t (Tensor): time points in which to integrate in descending interval [1, 0], size (N)
            log_p0 (Callable[[Tensor], Tensor]): function that calculates log likelihood
                at t=0 given points at t=0
            method (str, optional):method which to use for integration.
                Check torchdiffeq odeint for all possibilities.
                Defaults to "midpoint".
            step_size (Optional[float], optional): if the method allows for a step size,
                the step size to use. Check torchdiffeq odein if possible valid
                Defaults to None.

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
        def dynamics_eq(
            t: Tensor, states: tuple[Tensor, Tensor]
        ) -> tuple[Tensor, Tensor]:
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

            return ut.detach(), div.detach()

        # the initial states of the dynamics eq are:
        # x at t=1 for position and the 0 vector for divergence
        init_states = (x1, torch.zeros(x1.shape[0], device=x1.device))
        ode_options = {"step_size": step_size} if step_size else {}

        with torch.no_grad():
            sol, div = odeint(
                dynamics_eq, init_states, t, method=method, options=ode_options
            )

        x0 = sol[-1]
        log_p_x0 = log_p0(x0)

        return x0, log_p_x0 + div[-1]

    def compute_likelihood(
        self,
        x1: Tensor,
        t: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        method: str = "midpoint",
        ode_step_size: Optional[float] = None,
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
            method (str, optional):method which to use for integration.
                Check torchdiffeq odeint for all possibilities.
                Defaults to "midpoint".
            step_size (Optional[float], optional): if the method allows for a step size,
                the step size to use. Check torchdiffeq odein if possible valid
                Defaults to None.
            est_steps (int, optional): number of estimations before aggregation. Defaults to 10.

        Returns:
            tuple[Tensor, Tensor]: a (B, D) tensor of the positions at t=0
            and a (B,1) tensor of likelihoods for those positions
        """

        log_p = 0.0
        sol = 0.0

        for _ in range(est_steps):
            sol_est, log_p_est = self.compute_likelihood_once(
                x1, t, log_p0, method, ode_step_size, **vf_extras
            )

            log_p += log_p_est
            sol += sol_est

        return sol / est_steps, log_p / est_steps  # type: ignore

    def classify(
        self,
        x: Tensor,
        anc_ts: Tensor,
        method: str = "midpoint",
        ode_step_size: Optional[float] = None,
        est_steps: int = 1,
        **vf_extras
    ) -> tuple[Tensor, Tensor]:
        """Finds the most likely classes for the batch of inputs based on their
        divergencies at anchor times

        Args:
            x (Tensor): inputs to classify, size (B, D...)
            anc_ts (Tensor): anchor times of size (N,) in descending order,
                where the unknown distributions are in time
            method (str, optional): which method to use for divergence integration.
                Defaults to "midpoint".
            ode_step_size (Optional[float], optional): step size of the integrator for divergence.
                Defaults to None.
            est_steps (int, optional): how many times to estimate divergence before averaging it.
                Defaults to 1

        Returns:
            tuple[Tensor, Tensor]: divergencies summed for each time query
                in ascending anchor time, size (B, N) and classes for each of the inputs, size (B,)
        """

        # Rademacher distribution sampling for Hutchinson's
        z = (torch.randn_like(x) < 0) * 2.0 - 1.0

        # velocity DE
        def diff_eq(t: Tensor, x: Tensor) -> Tensor:
            return self.vector_field(x, t, **vf_extras)

        # dynamics system DE
        def dynamics_eq(
            t: Tensor, states: tuple[Tensor, Tensor]
        ) -> tuple[Tensor, Tensor]:
            xt, _ = states

            with torch.set_grad_enabled(True):
                xt.requires_grad_()  # needed xt grads for Hutchinson's
                ut = diff_eq(t, xt)

                ut_dot_z = torch.einsum(
                    "nm,nm->n", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_ut_dot_z = gradient(ut_dot_z, xt)

                div = torch.einsum(
                    "nm,nm->n",
                    grad_ut_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

            return ut.detach(), div.detach()

        # stuff for odeint
        init_states = (x, torch.zeros(x.shape[0], device=x.device))
        ode_options = {"step_size": ode_step_size} if ode_step_size else {}

        # matrix of divergencies from which classes will be drawn
        div_mat = torch.zeros(
            (x.shape[0], anc_ts.shape[0], anc_ts.shape[0]), device=x.device
        )

        # make for loop that expands the time interval for the dynamic system eq
        for t_key in range(2, anc_ts.shape[0] + 1):
            divs = 0  # type: ignore
            for _ in range(est_steps):
                _, est_divs = odeint(  # (anchors, B)
                    dynamics_eq,
                    init_states,
                    anc_ts[:t_key],
                    method=method,
                    options=ode_options,
                )  # type: ignore

                divs += est_divs  # type: ignore
            divs: Tensor = divs / est_steps

            # (B, anchors)
            divs: Tensor = torch.transpose(divs, 1, 0)

            # flip the anchors dims, so that it goes in ascending time order
            divs = divs.flip(1)

            # t_key row gets +divs for all the query times
            div_mat[:, t_key - 1, :t_key] += divs

            # t_key col gets -divs for all query times (reciprocal of divs)
            div_mat[:, :t_key, t_key - 1] -= divs

        # can maybe get away with just looking at the lower triangular
        div_mat = div_mat.sum(dim=-1)
        preds = torch.argmax(div_mat, -1)

        return div_mat, preds
