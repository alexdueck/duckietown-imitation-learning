"""OpenGL compatibility helpers for gym-duckietown 6.2.0."""

from __future__ import annotations


def prepare_reset_render_context(env) -> bool:
    """Make resets independent of the previously rendered camera view.

    Simulator.reset() configures GL_LIGHT0 before selecting its shadow context or
    resetting the model-view matrix. OpenGL transforms the light position using
    that stale matrix, so GUI event timing can otherwise change observations.
    """
    raw_env = getattr(env, "unwrapped", env)
    shadow_window = getattr(raw_env, "shadow_window", None)
    if shadow_window is None:
        return False

    from pyglet import gl

    shadow_window.switch_to()
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    return True
