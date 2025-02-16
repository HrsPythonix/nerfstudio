# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for camera paths.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length


def get_interpolated_camera_path(cameras: Cameras, steps: int) -> Cameras:
    """Generate a camera path between two cameras.

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices().cpu().numpy()
    poses = cameras.camera_to_worlds.cpu().numpy()
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)

    cameras = Cameras(fx=Ks[:, 0, 0], fy=Ks[:, 1, 1], cx=Ks[0, 0, 2], cy=Ks[0, 1, 2], camera_to_worlds=poses)
    return cameras


def get_spiral_path(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a spiral trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)

    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
    )


def get_task_path(
    device: Union[torch.device, str],
    cam_positions: list,
    cam_rotations: list,
    render_width: int,
    render_height: int,
    fov: float,
) -> Cameras:
    fovx = math.radians(fov)
    cx = render_width // 2
    cy = render_height // 2
    fx = cx / math.tan(fovx / 2)
    fy = fx

    cx = torch.tensor([cx], device=device)
    cy = torch.tensor([cy], device=device)
    fx = torch.tensor([fx], device=device)
    fy = torch.tensor([fy], device=device)

    up_vec = torch.tensor([0.0, 0.0, 1.0], device=device)

    c2whs = []
    for pos, rot in zip(cam_positions, cam_rotations):
        camera_pos = torch.tensor(pos, device=device)
        # lookat = camera_pos - center
        lookat = [-math.cos(rot[0]) * math.cos(rot[1]), -math.cos(rot[0]) * math.sin(rot[1]), -math.sin(rot[0])]
        lookat = torch.tensor(lookat, device=device, dtype=torch.float)
        c2w = camera_utils.viewmatrix(lookat, up_vec, camera_pos)
        c2wh = pose_utils.to4x4(c2w)
        c2whs.append(c2wh[:3, :4])
    c2whs = torch.stack(c2whs, dim=0)

    return Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2whs)


def get_circle_path(
    camera: Cameras,
    center: Optional[torch.tensor] = None,
    steps: int = 120,
    radius: float = 0.6,
    up_vec: Optional[torch.tensor] = None,
    height: Optional[float] = None,
    render_width: Optional[torch.tensor] = None,
    render_height: Optional[torch.tensor] = None,
) -> Cameras:
    if center is None:
        center = torch.tensor([0.0, 0.0, 0.0], device=camera.device)
    if up_vec is None:
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=camera.device)
    if height is None:
        height = 0.0
    if render_width is None:
        render_width = torch.tensor([1920], device=camera.device)
    if render_height is None:
        render_height = torch.tensor([1080], device=camera.device)

    c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi, steps + 1)[:-1]:
        camera_pos = torch.tensor([torch.cos(theta), torch.sin(theta), height], device=camera.device) * radius
        lookat = camera_pos - center
        c2w = camera_utils.viewmatrix(lookat, up_vec, camera_pos)
        c2wh = pose_utils.to4x4(c2w)
        c2whs.append(c2wh[:3, :4])
    c2whs = torch.stack(c2whs, dim=0)
    return Cameras(fx=camera.fx[0], fy=camera.fy[0], cx=render_width / 2, cy=render_height / 2, camera_to_worlds=c2whs)


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]
    if "camera_type" in camera_path:
        camera_type = camera_path["camera_type"]
    else:
        camera_type = "perspective"

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    else:
        camera_type = CameraType.PERSPECTIVE

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2ws.append(c2w)
        if camera_type == CameraType.EQUIRECTANGULAR:
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fxs.append(focal_length)
            fys.append(focal_length)

    # Iff ALL cameras in the path have a "time" value, construct Cameras with times
    if all("render_time" in camera for camera in camera_path["camera_path"]):
        times = torch.tensor([camera["render_time"] for camera in camera_path["camera_path"]])
    else:
        times = None

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        camera_to_worlds=camera_to_worlds,
        camera_type=camera_type,
        times=times,
    )
