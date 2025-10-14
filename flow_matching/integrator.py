from typing import Callable
import torch
from torch import Tensor


class Integrator:
    """Base class for integrating an ODE in time given some starting conditon"""

    def _step(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        tn: Tensor,
        xn: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        func: Callable[[Tensor, Tensor], Tensor],
        x0: Tensor,
        ints: Tensor,
        steps: int = 20,
    ) -> tuple[Tensor, Tensor]:
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

        x = torch.empty(size=(steps + 1, *x0.shape), dtype=x0.dtype, device=x0.device)
        t = torch.empty(
            size=(steps + 1, *t0.shape), dtype=ints.dtype, device=ints.device
        )

        x[0] = x0
        t[0] = t0

        xn = x0
        tn = t0
        for e in range(steps):
            tn, xn = self._step(func, tn, xn, dt)

            x[e + 1] = xn
            t[e + 1] = tn

        return t, x


class EulerIntegrator(Integrator):
    """
    Euler ODE solver.

    For a function dx/dt = f(t, x) it calculates the solution numerically as:
    xn+1 = xn + dt * f(tn, xn)
    """

    def _step(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        tn: Tensor,
        xn: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        xn = xn + dt * func(tn, xn)
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
        func: Callable[[Tensor, Tensor], Tensor],
        tn: Tensor,
        xn: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        half_dt = dt / 2
        mid_x = xn + half_dt * func(tn, xn)

        xn = xn + dt * func(tn + half_dt, mid_x)
        tn = tn + dt

        return tn, xn


def main():
    # always define dy/dt as f(t, y)
    f = lambda t, x: 2 * t

    x0 = torch.zeros((10, 2))
    tint = torch.tensor([[0.0, 1.0]], dtype=x0.dtype, device=x0.device)
    tint = tint.expand(*x0.shape)

    ei = EulerIntegrator()
    t, x = ei.integrate(f, x0, tint, steps=4)
    print(t[-1], x[-1])

    mi = MidpointIntegrator()
    t, x = mi.integrate(f, x0, tint, steps=4)
    print(t[-1], x[-1])


if __name__ == "__main__":
    main()
