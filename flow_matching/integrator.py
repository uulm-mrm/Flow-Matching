from typing import Callable
import torch
from torch import Tensor


class Integrator:
    """Base class for integrating an ODE in time given some starting conditon"""

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Does one step of the numerical ODE solver

        Args:
            func (Callable[[Tensor, Tensor], Tensor]): f from dx/dt = f(t, x)
            tn (Tensor): current time, size (B,)
            xn (Tensor): current solution, size (B, D...)
            dt (Tensor): delta time increment, size (B, )

        Returns:
            tuple[Tensor, Tensor]: next iteration time and solution
        """
        raise NotImplementedError

    def integrate(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        x0: list[Tensor],
        ints: Tensor,
        steps: int = 20,
    ) -> tuple[Tensor, list[Tensor]]:
        """Integrates the function `func` over a batch of time intervals `ints`,
        starting from some initial condition `x0`, iterating `steps` number of times

        Args:
            func (Callable[[Tensor, Tensor], Tensor]): dx/dt = f(t, x) ODE to solve
            x0 (Tensor): batch of initial conditions, size (B, D...)
            ints (Tensor): batch of intervals in ascending or descending orders, size (B, 2)
            steps (int, optional): number of steps to numerically solve for. Defaults to 20.

        Returns:
            tuple[Tensor, Tensor]: trajectory of time and solutions given the parameters,
            size (steps+1, B) for time and (steps+1, B, D...) for solutions
        """
        t0, t1 = ints[:, 0].unsqueeze(1), ints[:, 1].unsqueeze(1)
        dt = (t1 - t0) / steps

        x_traj = [
            torch.empty(
                size=(steps + 1, *_x0.shape), dtype=_x0.dtype, device=_x0.device
            )
            for _x0 in x0
        ]

        t_traj = torch.empty(
            size=(steps + 1, *t0.shape), dtype=ints.dtype, device=ints.device
        )

        # initial state
        for i, _x0 in enumerate(x0):
            x_traj[i][0] = _x0

        t_traj[0] = t0

        xn = x0
        tn = t0
        for e in range(steps):
            tn, xn = self._step(func, tn, xn, dt)

            for i, _xn in enumerate(xn):
                x_traj[i][e + 1] = _xn
            t_traj[e + 1] = tn

        return t_traj, x_traj


class EulerIntegrator(Integrator):
    """
    Euler ODE solver.

    For a function dx/dt = f(t, x) it calculates the solution numerically as:
    xn+1 = xn + dt * f(tn, xn)
    """

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        states = func(tn, xn)

        for i, state in enumerate(states):
            xn[i] = xn[i] + dt * state

        tn = tn + dt

        return tn, xn


class MidpointIntegrator(Integrator):
    """
    Explicit Extended Euler ODE Solver (Midpoint Solver)

    For a function dx/dt = f(t, x) it calculates the solution numerically as:
    xn+1 = xn + dt * f(tn + dt / 2, xn + dt / 2 * f(tn, xn))
    """

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        half_dt = dt / 2

        mid_states = func(tn, xn)
        for i, state in enumerate(mid_states):
            mid_states[i] = xn[i] + half_dt * state

        states = func(tn + half_dt, mid_states)
        for i, state in enumerate(states):
            xn[i] = xn[i] + dt * state

        tn = tn + dt

        return tn, xn


def main():
    # always define dy/dt as f(t, y)
    f = lambda t, x: [t, 2 * t]

    batch_size = 10
    x0 = [torch.zeros((batch_size, 2)), torch.zeros((batch_size, 2))]
    tint = torch.tensor([[0.0, 1.0]])
    tint = tint.expand(batch_size, 2)

    # ei = EulerIntegrator()
    # t, x = ei.integrate(f, x0, tint, steps=4)
    # x1, x2 = x
    # print(t[-1], x1[-1], x2[-1])

    mi = MidpointIntegrator()
    t, x = mi.integrate(f, x0, tint, steps=10)
    x1, x2 = x
    print(t[-1], x1[-1], x2[-1])


if __name__ == "__main__":
    main()
