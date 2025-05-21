import numpy as np
import plotly.graph_objects as go


def plot_riemann_sphere(show=True, save_path=None):
    """Generate a Riemann sphere with symbolic anchors and Möbius-like loops."""
    # Sphere surface
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)
    theta, phi = np.meshgrid(theta, phi)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    surface = go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.3, showscale=False)

    # Anchor points representing five aspects of self
    anchors = {
        "Ka": (1, 0, 0),
        "Ba": (-1, 0, 0),
        "Akh": (0, 1, 0),
        "Ren": (0, -1, 0),
        "Šwt": (0, 0, 1),
    }
    anchor_scatter = go.Scatter3d(
        x=[p[0] for p in anchors.values()],
        y=[p[1] for p in anchors.values()],
        z=[p[2] for p in anchors.values()],
        mode="markers+text",
        text=list(anchors.keys()),
        textposition="top center",
        marker=dict(size=5, color="red"),
    )

    # Möbius-like loop: a simple twisted circle around the sphere
    t = np.linspace(0, 2 * np.pi, 200)
    r = 0.7
    loop_x = r * np.cos(t)
    loop_y = r * np.sin(t)
    loop_z = 0.2 * np.sin(2 * t)
    loop = go.Scatter3d(x=loop_x, y=loop_y, z=loop_z, mode="lines", line=dict(color="purple"))

    fig = go.Figure(data=[surface, anchor_scatter, loop])
    fig.update_layout(scene=dict(aspectmode="data"))
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a cognitive Riemann sphere")
    parser.add_argument("--output", help="Path to save HTML visualization", default="")
    args = parser.parse_args()

    path = args.output if args.output else None
    plot_riemann_sphere(show=not path, save_path=path)
