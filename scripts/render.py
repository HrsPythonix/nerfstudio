#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torchtyping import TensorType
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import (
    get_circle_path,
    get_path_from_json,
    get_spiral_path,
    get_task_path,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    camera_type: CameraType = CameraType.PERSPECTIVE,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        camera_type: Camera projection format type.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    if output_format == "images":
        output_image_dir = output_filename.parent / output_filename.stem
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):

                aabb_box = None
                if crop_data is not None:
                    bounding_box_min = crop_data.center - crop_data.scale / 2.0
                    bounding_box_max = crop_data.center + crop_data.scale / 2.0
                    aabb_box = SceneBox(torch.stack([bounding_box_min, bounding_box_max]).to(pipeline.device))
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    if output_image.shape[-1] == 1:
                        output_image = np.concatenate((output_image,) * 3, axis=-1)
                    render_image.append(output_image)
                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    if output_format == "video":
        if camera_type == CameraType.EQUIRECTANGULAR:
            insert_spherical_metadata_into_file(output_filename)


def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16
    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()


def render_task(cameras: Cameras, save_list: List[str], pipeline: Pipeline, rendered_output_names: List[str]):
    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )

    with progress:
        for cam_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=cam_idx)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        return
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    render_image.append(output_image)
                render_image = np.concatenate(render_image, axis=1)
                media.write_image(save_list[cam_idx], render_image)


def UE_coord_to_NeRF(X: float, Y: float, Z: float):
    return [X / 400.0, Y / 400.0, Z / 400.0 - 0.3]


def parse_task_json(task_json_path: str, pipeline: Pipeline):
    ref_cam = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
    with open(task_json_path, "r") as f:
        task = json.load(f)

        fov = float(task["CaptureSetting"]["Fov"])
        render_width = int(task["CaptureSetting"]["ImageWidth"])
        render_height = int(task["CaptureSetting"]["ImageHeight"])
        root_path = task["CaptureSetting"]["TargetDirectory"]

        capture_position_list = []
        capture_rotation_list = []
        capture_save_list = []
        for cap in task["CapturePosInfo"]:
            capture_save_list.append(os.path.join(root_path, cap["SavePath"]))
            capture_position_list.append(
                UE_coord_to_NeRF(float(cap["Loc"]["X"]), float(cap["Loc"]["Y"]), float(cap["Loc"]["Z"]))
            )
            capture_rotation_list.append(
                [float(cap["Rot"]["Pitch"]), float(cap["Rot"]["Yaw"]), float(cap["Rot"]["Roll"])]
            )

        camera = get_task_path(ref_cam, capture_position_list, capture_rotation_list, render_width, render_height, fov)

        return camera, capture_save_list


def start_server(
    pipeline: Pipeline,
    task_dir: str,
    log_path: str,
    query_interval: float,
    rendered_output_names: List[str],
    remove_after_parse: bool,
):
    json_postfix = "*.json"
    while True:
        time.sleep(query_interval)

        task_list = glob.glob(os.path.join(task_dir, json_postfix))
        if len(task_list) == 0:
            CONSOLE.print("No task detected...")
            continue
        else:
            CONSOLE.print("Task detected!")
            cameras = save_list = None
            try:
                cameras, save_list = parse_task_json(task_list[0], pipeline)
                CONSOLE.print("Task %s json parsed!" % (os.path.basename(task_list[0])))
            except Exception as e:
                traceback.print_exc()
                print(e)
                CONSOLE.print("Task %s json parse failed!" % (os.path.basename(task_list[0])))
                continue

            if remove_after_parse:
                os.remove(task_list[0])
                CONSOLE.print("Task %s json deleted!" % (os.path.basename(task_list[0])))

            with open(log_path, "w") as f:
                f.write("running, %s" % (os.path.basename(task_list[0])))
                CONSOLE.print("Task %s running log dumped!" % (os.path.basename(task_list[0])))

            if cameras and save_list:
                try:
                    render_task(cameras, save_list, pipeline, rendered_output_names)
                    CONSOLE.print("Task %s rendered!" % (os.path.basename(task_list[0])))
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    continue

            with open(log_path, "w") as f:
                f.write("done, %s" % (os.path.basename(task_list[0])))
                CONSOLE.print("Task %s done log dumped!" % (os.path.basename(task_list[0])))


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: TensorType[3] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    center: TensorType[3] = torch.Tensor([0.0, 0.0, 0.0])
    """center of the crop"""
    scale: TensorType[3] = torch.Tensor([2.0, 2.0, 2.0])
    """scale of the crop"""


def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None

    bg_color = camera_json["crop"]["crop_bg_color"]

    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        center=torch.Tensor(camera_json["crop"]["crop_center"]),
        scale=torch.Tensor(camera_json["crop"]["crop_scale"]),
    )


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    load_config: Path
    """Path to config YAML file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "circle", "server", "filename", "train", "eval"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    """Scaling factor to apply to the camera image resolution."""
    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_path: Path = Path("renders/output.mp4")
    """Name of the output file."""
    seconds: float = 5.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    eval_num_rays_per_chunk: Optional[int] = None
    # camera circle center
    circle_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # camera circle radius
    circle_radius: Optional[float] = None
    # camera circle height
    circle_height: Optional[float] = None
    # camera circle up vector
    circle_up_vec: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # fps for output video
    fps: Optional[int] = None
    # output video resolution width
    render_width: Optional[int] = None
    # output video resolution height
    render_height: Optional[int] = None
    # task directory
    task_dir: Optional[str] = "/cfs/risheng/workspace/NeRF/task/"
    # log path
    log_path: Optional[str] = "/cfs/risheng/workspace/NeRF/log/server.log"
    # query interval
    query_interval: Optional[float] = 1.0
    # remove json after parse
    remove_after_parse: Optional[bool] = True

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" or "circle" or "server" else "inference",
        )

        if self.traj != "server":
            install_checks.check_ffmpeg_installed()

        seconds = self.seconds
        crop_data = None

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            # TODO(ethan): pass in the up direction of the camera
            camera_type = CameraType.PERSPECTIVE
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "train":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            camera_path = pipeline.datamanager.train_dataset.cameras.to(camera_start.device)
            camera_type = pipeline.datamanager.train_dataset.cameras.camera_type
        elif self.traj == "eval":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            camera_path = pipeline.datamanager.eval_dataset.cameras.to(camera_start.device)
            camera_type = pipeline.datamanager.eval_dataset.cameras.camera_type
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            if "camera_type" not in camera_path:
                camera_type = CameraType.PERSPECTIVE
            elif camera_path["camera_type"] == "fisheye":
                camera_type = CameraType.FISHEYE
            elif camera_path["camera_type"] == "equirectangular":
                camera_type = CameraType.EQUIRECTANGULAR
            else:
                camera_type = CameraType.PERSPECTIVE
            crop_data = get_crop_from_json(camera_path)
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            camera_type=camera_type,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
