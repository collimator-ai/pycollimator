import numpy as np
import jax.numpy as jnp

from ...framework import LeafSystem


class PlanarQuadrotor(LeafSystem):
    def __init__(self, *args, m=1.0, I_B=1.0, r=0.5, g=9.81, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_parameter("m", m)
        self.declare_parameter("I_B", I_B)
        self.declare_parameter("r", r)
        self.declare_parameter("g", g)

        self.declare_input_port(name="u")

        self.declare_continuous_state(shape=(6,), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y, θ, dx, dy, dθ = state.continuous_state
        (u,) = inputs

        m = parameters["m"]
        I_B = parameters["I_B"]
        r = parameters["r"]
        g = parameters["g"]

        ddx = -(1 / m) * (u[0] + u[1]) * jnp.sin(θ)
        ddy = (1 / m) * (u[0] + u[1]) * jnp.cos(θ) - g
        ddθ = (1 / I_B) * r * (u[0] - u[1])

        return jnp.array([dx, dy, dθ, ddx, ddy, ddθ])

    def animate(
        self,
        xf,
        xlim=None,
        ylim=None,
        figsize=(4, 4),
        interval=50,
        stride=1,
        notebook=True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython import display

        xf = xf[:, ::stride]

        xc, yc, θc = xf[0], xf[1], xf[2]

        fig, ax = plt.subplots(figsize=figsize)

        hx, hy = 0.4, 0.1
        r2d = 180 / np.pi
        body = plt.Rectangle(
            (0, 0),
            hx,
            hy,
            angle=0,
            rotation_point="center",
            fc="xkcd:light grey",
            ec="xkcd:light grey",
        )

        ax.add_patch(body)

        if xlim is None:
            xlim = [-1.5, 1.5]
        if ylim is None:
            ylim = [-2, 2]

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid()

        def _animate(i):
            # First set xy to be the center, then rotate, then move xy to bottom left
            body.set_xy((xc[i], yc[i]))
            body.angle = θc[i] * r2d
            body.set_xy((xc[i] - hx / 2, yc[i] - hy / 2))

            return (body,)

        anim = FuncAnimation(fig, _animate, frames=xf.shape[1], interval=interval)

        if not notebook:
            return anim

        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()  # avoid plotting a spare static plot
