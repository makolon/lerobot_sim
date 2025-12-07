"""LeRobot SO ARM100 base task which uses a MuJoCo robot model."""

import collections
from collections.abc import Mapping
import copy
import dataclasses
import enum
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer import initializers
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import variation_broadcaster
from dm_env import specs
import immutabledict
import numpy as np
from numpy import typing as npt


# SO ARM100 HOME positions: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
# Based on the "home" keyframe in so_arm100.xml
HOME_CTRL: npt.NDArray[float] = np.array(
    [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
)
HOME_CTRL.setflags(write=False)
# SO ARM100 HOME qpos includes gripper as single joint (not mirrored)
HOME_QPOS: npt.NDArray[float] = np.array(
    [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
)
HOME_QPOS.setflags(write=False)


# The linear displacement that corresponds to fully open and closed gripper
# in sim. Note that the sim model does not model the dynamixel values, but
# rather the linear displacement of the fingers in meters.

# SIM_GRIPPER_CTRL_CLOSE controls the range of ctrl values that can be set,
# and is lower than the achievable qpos for the gripper, so that the
# proportional actuator can apply a force when the gripper is in the closed
# position.
# SIM_GRIPPER_QPOS_CLOSE is the value of qpos when the gripper is closed in
# sim.
# SO ARM100 gripper (Jaw) joint range: -0.174 (closed) to 1.75 (open) radians
# This is the qpos value of the gripper joint in MuJoCo simulation
SIM_GRIPPER_QPOS_OPEN: float = 1.75
SIM_GRIPPER_QPOS_CLOSE: float = -0.174

# Range used for setting ctrl
SIM_GRIPPER_CTRL_OPEN: float = 1.75
SIM_GRIPPER_CTRL_CLOSE: float = -0.174

# These are follower dynamixel values for OPEN and CLOSED gripper.
FOLLOWER_GRIPPER_OPEN: float = 1.5155
FOLLOWER_GRIPPER_CLOSE: float = -0.06135

LEADER_GRIPPER_OPEN: float = 0.78
LEADER_GRIPPER_CLOSE: float = -0.04

WRIST_CAMERA_POSITION: tuple[float, float, float] = (
    -0.011,
    -0.0814748,
    -0.0095955,
)


@dataclasses.dataclass(frozen=True)
class GripperLimit:
    """Gripper open and close limit.

    Attributes:
        open: Joint position of gripper being open.
        close: Joint position of gripper being closed.
    """

    open: float
    close: float


GRIPPER_LIMITS = immutabledict.immutabledict({
    'sim_qpos': GripperLimit(
        open=SIM_GRIPPER_QPOS_OPEN,
        close=SIM_GRIPPER_QPOS_CLOSE,
    ),
    'sim_ctrl': GripperLimit(
        open=SIM_GRIPPER_CTRL_OPEN,
        close=SIM_GRIPPER_CTRL_CLOSE,
    ),
    'follower': GripperLimit(
        open=FOLLOWER_GRIPPER_OPEN,
        close=FOLLOWER_GRIPPER_CLOSE,
    ),
    'leader': GripperLimit(
        open=LEADER_GRIPPER_OPEN,
        close=LEADER_GRIPPER_CLOSE,
    ),
})


_DEFAULT_PHYSICS_DELAY_SECS: float = 0.3
_DEFAULT_JOINT_OBSERVATION_DELAY_SECS: float = 0.1

# SO ARM100 joint names from the MJCF model
_ALL_JOINTS: tuple[str, ...] = (
    'Rotation',      # Base rotation (yaw)
    'Pitch',         # Shoulder pitch
    'Elbow',         # Elbow joint
    'Wrist_Pitch',   # Wrist pitch
    'Wrist_Roll',    # Wrist roll
    'Jaw',           # Gripper jaw
)


class GeomGroup(enum.IntFlag):
    NONE = 0
    ARM = enum.auto()
    GRIPPER = enum.auto()
    TABLE = enum.auto()
    OBJECT = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()


class LeRobotTask(composer.Task):
    """The base SO ARM100 task for single-arm manipulation."""

    def __init__(
        self,
        control_timestep: float,
        cameras: tuple[str, ...] = ('overhead_cam',),
        camera_resolution: tuple[int, int] = (480, 640),
        joints_observation_delay_secs: (
            variation.Variation | float
        ) = _DEFAULT_JOINT_OBSERVATION_DELAY_SECS,
        image_observation_enabled: bool = True,
        image_observation_delay_secs: (
            variation.Variation | float
        ) = _DEFAULT_PHYSICS_DELAY_SECS,
        update_interval: int = 1,
        waist_joint_limit: float = np.pi / 2,
        terminate_episode=True,
        mjcf_root: str | None = None,
    ):
        """Initializes a new SO ARM100 task.

        Args:
            control_timestep: Float specifying the control timestep in seconds.
            cameras: The default cameras to use.
            camera_resolution: The camera resolution to use for rendering.
            joints_observation_delay_secs: The delay of the joints observation. This
                can be a number or a composer.Variation. If set, also adds
                `undelayed_joints_pos` and `undelayed_joints_vel` observables for
                debugging.
            image_observation_enabled: Whether to enable physics state observation, as
                defined by `physics.get_state()`.
            image_observation_delay_secs: The delay of the `delayed_physics_state`
                observable. Note that the `physics_state` observable is not delayed.
                This can be a number or a composer.Variation. When set this also delays
                the camera observations.
            update_interval: An integer, number of simulation steps between successive
                updates to the value of this observable.
            waist_joint_limit: The joint limit for the waist joint, in radians. Only
                affects the action spec.
            terminate_episode: Whether to terminate the episode when the task
                succeeds.
            mjcf_root: The path to the scene XML file.
        """

        self._waist_joint_limit = waist_joint_limit
        self._terminate_episode = terminate_episode

        self._scene = Arena(
            camera_resolution=camera_resolution,
            mjcf_root_path=mjcf_root,
        )
        self._scene.mjcf_model.option.flag.multiccd = 'enable'
        self._scene.mjcf_model.option.noslip_iterations = 0

        self.control_timestep = control_timestep

        self._joints = [
            self._scene.mjcf_model.find('joint', name) for name in _ALL_JOINTS
        ]

        # Add custom camera observable.
        obs_dict = collections.OrderedDict()

        shared_delay = variation_broadcaster.VariationBroadcaster(
            image_observation_delay_secs / self.physics_timestep
        )
        cameras_entities = [
            self.root_entity.mjcf_model.find('camera', name) for name in cameras
        ]
        for camera_entity in cameras_entities:
            obs_dict[camera_entity.name] = observable.MJCFCamera(
                camera_entity,
                height=camera_resolution[0],
                width=camera_resolution[1],
                update_interval=update_interval,
                buffer_size=1,
                delay=shared_delay.get_proxy(),
                aggregator=None,
                corruptor=None,
            )
            obs_dict[camera_entity.name].enabled = True

        lerobot_observables = LeRobotObservables(
            self.root_entity,
        )
        lerobot_observables.enable_all()
        obs_dict.update(lerobot_observables.as_dict())
        self._task_observables = obs_dict

        if joints_observation_delay_secs:
            self._task_observables['undelayed_joints_pos'] = copy.copy(
                self._task_observables['joints_pos']
            )
            self._task_observables['undelayed_joints_vel'] = copy.copy(
                self._task_observables['joints_vel']
            )
            self._task_observables['joints_pos'].configure(
                delay=joints_observation_delay_secs / self.physics_timestep
            )
            self._task_observables['joints_vel'].configure(
                delay=joints_observation_delay_secs / self.physics_timestep
            )
            self._task_observables['delayed_joints_pos'] = copy.copy(
                self._task_observables['joints_pos']
            )
            self._task_observables['delayed_joints_vel'] = copy.copy(
                self._task_observables['joints_vel']
            )

        self._task_observables['physics_state'].enabled = (
            image_observation_enabled
        )
        if image_observation_delay_secs:
            self._task_observables['delayed_physics_state'] = copy.copy(
                self._task_observables['physics_state']
            )
            self._task_observables['delayed_physics_state'].configure(
                delay=shared_delay.get_proxy(),
            )

        self._all_props = []
        self._all_prop_placers = []
        if self._all_prop_placers:
            self._all_prop_placers.append(
                initializers.PropPlacer(
                    props=self._all_props,
                    position=deterministic.Identity(),
                    ignore_collisions=True,  # Collisions already resolved.
                    settle_physics=True,
                )
            )

    @property
    def root_entity(self) -> composer.Entity:
        return self._scene

    @property
    def task_observables(self) -> Mapping[str, observable.Observable]:
        return dict(
            **self._task_observables
        )

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        # SO ARM100 single arm: 0-4: arm joints (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
        # 5: gripper (Jaw)
        minimum = physics.model.actuator_ctrlrange[:, 0].astype(np.float32)
        maximum = physics.model.actuator_ctrlrange[:, 1].astype(np.float32)
        
        # Apply waist (shoulder_pan) joint limit
        minimum[0] = -self._waist_joint_limit
        maximum[0] = self._waist_joint_limit

        # Gripper actions are never delta actions.
        minimum[5] = GRIPPER_LIMITS['follower'].close
        maximum[5] = GRIPPER_LIMITS['follower'].open

        return specs.BoundedArray(
            shape=(6,),
            dtype=np.float32,
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def convert_gripper(
        cls,
        gripper_value: npt.NDArray[float],
        from_name: str,
        to_name: str,
    ) -> float:
        from_limits = GRIPPER_LIMITS[from_name]
        to_limits = GRIPPER_LIMITS[to_name]
        return (gripper_value - from_limits.close) / (
            from_limits.open - from_limits.close
        ) * (to_limits.open - to_limits.close) + to_limits.close

    def before_step(
        self,
        physics: mjcf.Physics,
        action: npt.ArrayLike,
        random_state: np.random.RandomState,
    ) -> None:
        # SO ARM100 single arm: action is [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
        arm_joints = action[:5]
        gripper_action = action[5]

        # Set arm joint controls
        np.copyto(physics.data.ctrl[:5], arm_joints)

        # Handle the gripper action
        gripper_cmd = np.array([gripper_action])
        gripper_ctrl = LeRobotTask.convert_gripper(
            gripper_cmd, 'follower', 'sim_ctrl'
        )

        np.copyto(physics.data.ctrl[5:6], gripper_ctrl)

    def get_reward(self, physics: mjcf.Physics) -> float:
        return 0.0

    def get_discount(self, physics: mjcf.Physics) -> float:
        if self.should_terminate_episode(physics):
            return 0.0
        return 1.0

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        if self._terminate_episode:
            reward = self.get_reward(physics)
            if reward >= 1.0:
                return True
        return False

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        arm_joints_bound = physics.bind(self._joints)

        # SO ARM100 has single arm with 6 joints
        arm_joints_bound.qpos[:] = HOME_QPOS

        np.copyto(physics.data.ctrl, HOME_CTRL)

        for prop_placer in self._all_prop_placers:
            prop_placer(physics, random_state)


class LeRobotObservables(composer.Observables):
    """LeRobot single-arm observables."""

    def as_dict(
        self, fully_qualified: bool = False
    ) -> collections.OrderedDict[str, observable.Observable]:
        return super().as_dict(fully_qualified=fully_qualified)

    @define.observable
    def joints_pos(self) -> observable.Observable:
        def _get_joints_pos(physics):
            # SO ARM100: 5 arm joints + 1 gripper joint
            gripper_pos = physics.data.qpos[5]

            gripper_qpos = LeRobotTask.convert_gripper(
                gripper_pos, 'sim_qpos', 'follower'
            )

            return np.concatenate([
                physics.data.qpos[:5],
                [gripper_qpos],
            ])
        return observable.Generic(_get_joints_pos)

    @define.observable
    def commanded_joints_pos(self) -> observable.Observable:
        """Returns commanded joint positions, for delta and absolute actions."""
        def _get_joints_cmd(physics):
            # Convert sim ctrl values to the environment-level actions
            gripper_ctrl = physics.data.ctrl[5]

            gripper_cmd = LeRobotTask.convert_gripper(
                gripper_ctrl,
                'sim_ctrl',
                'follower',
            )

            return np.concatenate([
                physics.data.ctrl[:5],
                [gripper_cmd],
            ])
        return observable.Generic(_get_joints_cmd)

    @define.observable
    def joints_vel(self) -> observable.Observable:
        return observable.MJCFFeature(
            'qvel',
            [self._entity.mjcf_model.find('joint', name) for name in _ALL_JOINTS],
        )

    @define.observable
    def physics_state(self) -> observable.Observable:
        return observable.Generic(lambda physics: physics.get_state())


class Arena(composer.Arena):
    """Standard Arena for SO ARM100.

    Forked from dm_control/manipulation/shared/arenas.py
    """

    def __init__(
        self,
        *args,
        camera_resolution,
        mjcf_root_path: str | None = None,
        **kwargs,
    ):
        self._camera_resolution = camera_resolution
        self.textures = []
        self._mjcf_root_path = mjcf_root_path
        super().__init__(*args, **kwargs)

    def _build(self, name: str | None = None) -> None:
        """Initializes this arena.

        Args:
            name: (optional) A string, the name of this arena. If `None`, use the
                model name defined in the MJCF file.
        """
        if not self._mjcf_root_path:
            self._mjcf_root_path = os.path.join(
                os.path.dirname(__file__),
                '../assets',
                'so_arm100/scene.xml',
            )

        self._mjcf_root = mjcf.from_path(
            path=self._mjcf_root_path,
            escape_separators=True,
        )
        self._mjcf_root.visual.__getattr__('global').offheight = (
            self._camera_resolution[0]
        )
        self._mjcf_root.visual.__getattr__('global').offwidth = (
            self._camera_resolution[1]
        )