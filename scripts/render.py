#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import os
import glob
import sys
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple

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
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path, get_circle_path, get_task_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
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
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)
            if output_format == "images":
                media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
            else:
                images.append(render_image)

    if output_format == "video":
        fps = len(images) / seconds
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")

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
                        CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                        return
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    render_image.append(output_image)
                render_image = np.concatenate(render_image, axis=1)
                media.write_image(save_list[cam_idx], render_image)

def UE_coord_to_NeRF(X:float, Y:float, Z:float):
    return [X / 400.0, Y / 400.0, Z / 400.0 - 0.3]

def parse_task_json(task_json_path:str, pipeline: Pipeline):
    ref_cam = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
    with open(task_json_path, 'r') as f:
        task = json.load(f)

        fov = float(task['CaptureSetting']['Fov'])
        render_width = int(task['CaptureSetting']['ImageWidth'])
        render_height = int(task['CaptureSetting']['ImageHeight'])
        root_path = task['CaptureSetting']['TargetDirectory']

        capture_position_list = []
        capture_rotation_list = []
        capture_save_list = []
        for cap in task['CapturePosInfo']:
            capture_save_list.append(os.path.join(root_path, cap['SavePath']))
            capture_position_list.append(UE_coord_to_NeRF(cap['Loc']['X'], 
                                                          cap['Loc']['Y'], 
                                                          cap['Loc']['Z']))
            capture_rotation_list.append([cap['Rot']['Pitch'], cap['Rot']['Yaw'], cap['Rot']['Roll']])
        
        camera = get_task_path(ref_cam, capture_position_list, capture_rotation_list, render_width, render_height, fov)

        return camera, capture_save_list
        
def start_server(pipeline: Pipeline, task_dir:str, log_path:str, query_interval:float, rendered_output_names:List[str], remove_after_parse:bool):
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
                CONSOLE.print("Task %s json parsed!"%(os.path.basename(task_list[0])))
            except Exception as e:
                print(e)
                CONSOLE.print("Task %s json parse failed!"%(os.path.basename(task_list[0])))
                continue
            
            if remove_after_parse:
                os.remove(task_list[0])
                CONSOLE.print("Task %s json deleted!"%(os.path.basename(task_list[0])))

            with open(log_path, 'w') as f:
                f.write("running, %s"%(os.path.basename(task_list[0])))
                CONSOLE.print("Task %s running log dumped!"%(os.path.basename(task_list[0])))

            if cameras and save_list:
                try:
                    render_task(cameras, save_list, pipeline, rendered_output_names)
                    CONSOLE.print("Task %s rendered!"%(os.path.basename(task_list[0])))
                except Exception as e:
                    print(e)
                    continue
                
            with open(log_path, 'w') as f:
                f.write("done, %s"%(os.path.basename(task_list[0])))
                CONSOLE.print("Task %s done log dumped!"%(os.path.basename(task_list[0])))

@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "circle", "server", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Specifies number of rays per chunk during eval.
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
        
        if self.traj != "server":
            # TODO(ethan): use camera information from parsing args
            if self.traj == "spiral":
                camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
                # TODO(ethan): pass in the up direction of the camera
                camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
            elif self.traj == "circle":
                val_camera = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
                circle_center = torch.tensor(self.circle_center, device=val_camera.device) if self.circle_center else None
                circle_up_vec = torch.tensor(self.circle_up_vec, device=val_camera.device) if self.circle_up_vec else None
                render_width = torch.tensor(self.render_width, device=val_camera.device) if self.render_width else None
                render_height = torch.tensor(self.render_height, device=val_camera.device) if self.render_height else None
                frame_num = int(self.fps * seconds)
                camera_path = get_circle_path(val_camera, circle_center, frame_num, self.circle_radius, circle_up_vec, self.circle_height, render_width, render_height)
            elif self.traj == "filename":
                with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                    camera_path = json.load(f)
                seconds = camera_path["seconds"]
                camera_path = get_path_from_json(camera_path)
            else:
                assert_never(self.traj)

            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                seconds=seconds,
                output_format=self.output_format,
            )
        else:
            start_server(pipeline, 
                         self.task_dir, 
                         self.log_path, 
                         self.query_interval, 
                         self.rendered_output_names, 
                         self.remove_after_parse
                         )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
