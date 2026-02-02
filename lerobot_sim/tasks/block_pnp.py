"""Block Pick and Place task."""

import os

from lerobot_sim.tasks import lerobot_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np

_BLOCK_RESET_HEIGHT = 0.02
_TARGET_RADIUS = 0.05

# Blue block position (start)
blue_block_uniform_position = distributions.Uniform(
    low=[-0.15, -0.05, _BLOCK_RESET_HEIGHT],
    high=[-0.05, 0.05, _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Target position
TARGET_POS = np.array([0.15, 0.0, 0.0])

# Random rotation around Z-axis
block_z_rotation = rotations.UniformQuaternion()


class BlockPnP(lerobot_task.LeRobotTask):
    """Pick and place a blue block to a target position."""

    def __init__(
        self,
        blue_block_path: str | None = None,
        **kwargs,
    ):
        """Initializes a new `BlockPnP` task.

        Args:
            blue_block_path: Path to asset of the blue block.
            **kwargs: Additional args to pass to the base class.
        """
        super().__init__(**kwargs)

        assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
        
        # Default block path if not provided
        if blue_block_path is None:
            blue_block_path = os.path.join(assets_dir, 'blocks', 'blue_block.xml')

        # Load blue block
        self._blue_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(blue_block_path)
        )
        self._scene.add_free_entity(self._blue_block_prop)

        # Remove freejoint to use PropPlacer
        freejoint = traversal_utils.get_freejoint(
            self._blue_block_prop.mjcf_model.find_all('body')[0]
        )
        if freejoint:
            freejoint.remove()

        # Add target site
        self._scene.mjcf_model.worldbody.add(
            'site', 
            name='target', 
            pos=f"{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}", 
            size=f"{_TARGET_RADIUS} 0.001", 
            rgba="1 0 0 0.3", 
            type="cylinder"
        )

        # Create prop placer for block
        self._placer = initializers.PropPlacer(
            props=[self._blue_block_prop],
            position=blue_block_uniform_position,
            quaternion=block_z_rotation,
            ignore_collisions=True,
            settle_physics=True,
        )

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._placer(physics, random_state)

    def get_reward(self, physics):
        """Returns 1.0 if the blue block is on the ground within the target."""
        blue_body = self._blue_block_prop.mjcf_model.find_all('body')[0]
        blue_geom = self._blue_block_prop.mjcf_model.find_all('geom')[0]
        blue_bind = physics.bind(blue_body)

        blue_pos = blue_bind.xpos

        # Calculate distance to target (only xy plane)
        dist = np.linalg.norm(blue_pos[:2] - TARGET_POS[:2])

        # Require the block to be resting on the ground plane (z ~= half-height).
        half_height = float(blue_geom.size[2])
        ground_z = 0.0
        ground_tol = 0.005  # 5mm tolerance for contact/settling
        on_ground = abs(blue_pos[2] - (ground_z + half_height)) <= ground_tol

        # Return reward based on distance and ground contact.
        if dist < _TARGET_RADIUS and on_ground:
            return 1.0
        return 0.0
