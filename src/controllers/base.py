"""Controller base class (SPEC 1: reset/step/save/load)."""

import numpy as np


class Controller:
    name = "base"
    is_neural = False
    # Jacobian of the controller torque w.r.t. -qd (6x6 or None); the engine
    # treats it implicitly for numerical stability (see simulate.rollout).
    damping_matrix = None

    def reset(self):
        """Called at the start of every trial."""

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        """Return the commanded joint torque (6,) for this step.

        ctx provides (all measurable or engine-shared): dt, g, bias, M, p, J,
        step index k, tau_ff (reference feedforward), and tau_patient which is
        the interaction-torque sensor analogue (used only by admittance).
        """
        raise NotImplementedError

    def on_trial_end(self, rec):
        """Hook called once after each trial (used e.g. by ILC)."""

    def save(self, path):
        np.savez(path, name=self.name)

    def load(self, path):
        return self
