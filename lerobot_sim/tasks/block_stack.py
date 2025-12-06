"""Block stacking task."""

import copy
import os

from lerobot_sim.tasks import lerobot_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np


TABLE_HEIGHT = 0.0
_BLOCK_RESET_HEIGHT = 0.03


# Red block position (to be placed on top)
red_block_uniform_position = distributions.Uniform(
    low=[-0.15, 0.15, TABLE_HEIGHT + _BLOCK_RESET_HEIGHT],
    high=[-0.05, 0.25, TABLE_HEIGHT + _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Blue block position (base block - bottom)
blue_block_uniform_position = distributions.Uniform(
    low=[0.05, 0.15, TABLE_HEIGHT + _BLOCK_RESET_HEIGHT],
    high=[0.15, 0.25, TABLE_HEIGHT + _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Yellow block position (middle block)
yellow_block_uniform_position = distributions.Uniform(
    low=[-0.15, -0.05, TABLE_HEIGHT + _BLOCK_RESET_HEIGHT],
    high=[-0.05, 0.05, TABLE_HEIGHT + _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Random rotation around Z-axis
block_z_rotation = rotations.UniformQuaternion()


class BlockStack(lerobot_task.LeRobotTask):
    """Stack blocks in 3 levels: blue (bottom), yellow (middle), red (top)."""

    def __init__(
        self,
        red_block_path: str | None = None,
        blue_block_path: str | None = None,
        yellow_block_path: str | None = None,
        **kwargs,
    ):
        """Initializes a new `BlockStack` task.

        Args:
            red_block_path: Path to asset of the red block (top).
            blue_block_path: Path to asset of the blue block (bottom/base).
            yellow_block_path: Path to asset of the yellow block (middle).
            **kwargs: Additional args to pass to the base class.
        """
        super().__init__(
            **kwargs,
        )

        assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    
        # Default block paths if not provided
        if red_block_path is None:
            red_block_path = os.path.join(assets_dir, 'blocks', 'red_block.xml')
        if blue_block_path is None:
            blue_block_path = os.path.join(assets_dir, 'blocks', 'blue_block.xml')
        if yellow_block_path is None:
            yellow_block_path = os.path.join(assets_dir, 'blocks', 'yellow_block.xml')

        # Try to load blocks; skip if they don't load properly
        self._red_block_prop = None
        self._blue_block_prop = None
        self._yellow_block_prop = None
        self._block_placers = []
        
        # Load red block (top)
        self._red_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(red_block_path)
        )
        self._scene.add_free_entity(self._red_block_prop)

        # Load blue block (bottom)
        self._blue_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(blue_block_path)
        )
        self._scene.add_free_entity(self._blue_block_prop)

        # Load yellow block (middle)
        self._yellow_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(yellow_block_path)
        )
        self._scene.add_free_entity(self._yellow_block_prop)

        # Remove freejoints to use PropPlacer
        for prop in [self._red_block_prop, self._blue_block_prop, self._yellow_block_prop]:
            freejoint = traversal_utils.get_freejoint(
                prop.mjcf_model.find_all('body')[0]
            )
            if freejoint:
                freejoint.remove()

        # Adjust positions for table height offset
        red_block_position = copy.deepcopy(red_block_uniform_position)
        blue_block_position = copy.deepcopy(blue_block_uniform_position)
        yellow_block_position = copy.deepcopy(yellow_block_uniform_position)

        # Create prop placers for blocks (bottom to top order)
        self._block_placers = [
            initializers.PropPlacer(
                props=[self._blue_block_prop],
                position=blue_block_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                settle_physics=False,
                max_attempts_per_prop=100,
            ),
            initializers.PropPlacer(
                props=[self._yellow_block_prop],
                position=yellow_block_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                max_attempts_per_prop=100,
                settle_physics=False,
            ),
            initializers.PropPlacer(
                props=[self._red_block_prop],
                position=red_block_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                max_attempts_per_prop=100,
                settle_physics=False,
            ),
            initializers.PropPlacer(
                props=[self._red_block_prop, self._blue_block_prop, self._yellow_block_prop],
                position=deterministic.Identity(),
                quaternion=deterministic.Identity(),
                ignore_collisions=True,  # Collisions already resolved.
                settle_physics=True,
            ),
        ]

        # Update qpos for the three blocks (each has 7 DOF: 3 pos + 4 quat)
        extra_qpos = np.zeros((21,))

        scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
        if scene_key is not None:
            scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for prop_placer in self._block_placers:
            prop_placer(physics, random_state)

    def get_reward(self, physics):
        """Returns 1.0 if all 3 blocks are successfully stacked: blue (bottom), yellow (middle), red (top)."""
        # Get positions of all blocks
        red_block_pos = physics.bind(self._red_block_prop.mjcf_model.find_all('body')[0]).xpos
        blue_block_pos = physics.bind(self._blue_block_prop.mjcf_model.find_all('body')[0]).xpos
        yellow_block_pos = physics.bind(self._yellow_block_prop.mjcf_model.find_all('body')[0]).xpos
            
        # Block height assumption (in meters)
        block_height = 0.05
        horizontal_threshold = 0.05  # 5cm tolerance
        vertical_tolerance = 0.02  # ±2cm tolerance
        
        # Check if yellow block is on blue block (middle on bottom)
        horizontal_dist_yellow_blue = np.linalg.norm(yellow_block_pos[:2] - blue_block_pos[:2])
        vertical_dist_yellow_blue = yellow_block_pos[2] - blue_block_pos[2]
        yellow_on_blue = (
            horizontal_dist_yellow_blue < horizontal_threshold and
            block_height - vertical_tolerance < vertical_dist_yellow_blue < block_height + vertical_tolerance
        )
        
        # Check if red block is on yellow block (top on middle)
        horizontal_dist_red_yellow = np.linalg.norm(red_block_pos[:2] - yellow_block_pos[:2])
        vertical_dist_red_yellow = red_block_pos[2] - yellow_block_pos[2]
        red_on_yellow = (
            horizontal_dist_red_yellow < horizontal_threshold and
            block_height - vertical_tolerance < vertical_dist_red_yellow < block_height + vertical_tolerance
        )
        
        # Success: both conditions must be satisfied
        if yellow_on_blue and red_on_yellow:
            return 1.0
        
        # Partial rewards for intermediate progress
        if yellow_on_blue:
            return 0.5  # Yellow successfully placed on blue
        
        return 0.0