from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from dt_utils.gym_duckietown_rendering import prepare_reset_render_context


class GymDuckietownRenderingTests(unittest.TestCase):
    def test_prepare_reset_render_context_selects_shadow_and_resets_matrix(self) -> None:
        calls = Mock()
        shadow_window = Mock()
        shadow_window.switch_to = calls.switch_to
        gl = SimpleNamespace(
            GL_MODELVIEW=17,
            glMatrixMode=calls.glMatrixMode,
            glLoadIdentity=calls.glLoadIdentity,
        )
        pyglet = types.ModuleType("pyglet")
        pyglet.gl = gl
        env = SimpleNamespace(unwrapped=SimpleNamespace(shadow_window=shadow_window))

        with patch.dict(sys.modules, {"pyglet": pyglet}):
            applied = prepare_reset_render_context(env)

        self.assertTrue(applied)
        self.assertEqual(
            calls.mock_calls,
            [
                call.switch_to(),
                call.glMatrixMode(gl.GL_MODELVIEW),
                call.glLoadIdentity(),
            ],
        )

    def test_prepare_reset_render_context_allows_non_duckietown_env(self) -> None:
        self.assertFalse(prepare_reset_render_context(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
