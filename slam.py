import os
import sys, io, re
import time
from argparse import ArgumentParser
from datetime import datetime
import numpy as np

import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
from arguments import ModelHiddenParams
from argparse import ArgumentParser, Namespace

from ultralytics import YOLO
from contextlib import redirect_stdout, redirect_stderr


def merge_hparams(args, config):
    params = ["ModelHiddenParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args


class Tee(io.TextIOBase):
    def __init__(self, terminal_stream, file_stream):
        self.term = terminal_stream
        self.file = file_stream
        # pretty broad ANSI matcher (CSI ...)
        self._ansi = re.compile(r'\x1B\[[0-9;?]*[ -/]*[@-~]')

    def write(self, data):
        # print to terminal as-is (keep colors)
        self.term.write(data); self.term.flush()
        # write to file with colors stripped and \r -> \n for progress bars
        clean = self._ansi.sub('', data).replace('\r', '\n')
        self.file.write(clean); self.file.flush()
        return len(data)

    def flush(self):
        self.term.flush(); self.file.flush()

    def isatty(self):
        # make libs think this is a TTY so they keep coloring the terminal side
        return True


class SLAM:
    def __init__(self, config, save_dir=None, save_interval=None, load_path=None, iters=200, rigid_loss=4.0):
        #self.yolo_model = YOLO("yolov8s.pt")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        if config["model_params"]["dynamic_model"] == 'offset':
            config['Training']['dystart'] = 0

        self.config = config
        self.save_dir = save_dir

        # with open(os.path.join(self.save_dir, 'outputs.txt'), "w") as f, redirect_stdout(f), redirect_stderr(f):
        with open(os.path.join(self.save_dir, 'outputs.txt'), "w", buffering=1) as f:  # "a" to append; use "w" to overwrite
            tee = Tee(sys.stdout, f)
            # send BOTH stdout and stderr through tee
            with redirect_stdout(tee), redirect_stderr(tee):
        
                model_params = munchify(config["model_params"])
                opt_params = munchify(config["opt_params"])
                pipeline_params = munchify(config["pipeline_params"])
                self.model_params, self.opt_params, self.pipeline_params = (
                    model_params,
                    opt_params,
                    pipeline_params,
                )

                # self.live_mode = self.config["Dataset"]["type"] == "realsense"
                self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
                self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
                self.use_gui = self.config["Results"]["use_gui"]
                self.interp_type = self.config['model_params'].get('interp_type', 'linear')
                self.eval_rendering = self.config["Results"]["eval_rendering"]

                model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
                
                parser = ArgumentParser(description="Training script parameters")
                hp = ModelHiddenParams(parser)
                
                hp = merge_hparams(hp, self.config)

                self.dataset = load_dataset(
                    model_params, model_params.source_path, config=config
                )

                # self.dataset.num_imgs = 6
                
                self.gaussians = GaussianModel(model_params.sh_degree, config=self.config, args=hp, init_deform=config["model_params"]["dynamic_model"],
                                            nframes=len(self.dataset))
                self.gaussians.init_lr(6.0)
                
                #load the YOLO model
                if config['yolo']:
                    self.dataset.yolo_model = YOLO('pretrained/yolov9e-seg.pt')
                print("dataset length: ", len(self.dataset))
                
                if "bound" in self.config["Dataset"].keys():
                    xyz_max = self.config["Dataset"]["bound"][1]
                    xyz_min = self.config["Dataset"]["bound"][0]
                else:
                    xyz_max = [8, 8, 8] 
                    xyz_min = [-8, -8, -8]
                
                    
                bg_color = [1, 1, 1]
                self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                frontend_queue = mp.Queue()
                backend_queue = mp.Queue()

                q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
                q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

                self.config["Results"]["save_dir"] = save_dir
                self.config["Training"]["monocular"] = self.monocular

                self.frontend = FrontEnd(self.config)
                self.backend = BackEnd(self.config)

                if load_path is not None:
                    ply_path = os.path.join(load_path, "point_cloud", "final", "point_cloud.ply")
                    self.gaussians.load_ply(ply_path)
                    self.gaussians.init_deform = 'offset'
                    offsets_path = os.path.join(load_path, "offsets", "iteration_81500", "offsets.pth")
                    self.gaussians.load_offsets(offsets_path)
                    self.frontend.kf_indices = self.gaussians.kf_list
                    self.frontend.cameras = torch.load(os.path.join(load_path, 'frontend_cameras.pth'))
                    self.backend.gaussians = self.gaussians

                self.gaussians.training_setup(opt_params)
                if config["model_params"]["dynamic_model"] == 'mlp':
                    self.gaussians.deform.train_setting(hp)
                    self.gaussians.time_interval = 1/len(self.dataset)

                self.frontend.dataset = self.dataset
                self.frontend.background = self.background
                self.frontend.pipeline_params = self.pipeline_params
                self.frontend.frontend_queue = frontend_queue
                self.frontend.backend_queue = backend_queue
                self.frontend.q_main2vis = q_main2vis
                self.frontend.q_vis2main = q_vis2main
                self.frontend.dystart = config["Training"]["dystart"] if "dystart" in config["Training"].keys() else 0
                self.frontend.set_hyperparams()
                
                self.backend.dataset = self.dataset
                if load_path is None:
                    self.backend.gaussians = self.gaussians
                self.backend.background = self.background
                self.backend.cameras_extent = 6.0
                self.backend.pipeline_params = self.pipeline_params
                self.backend.opt_params = self.opt_params
                self.backend.frontend_queue = frontend_queue
                self.backend.backend_queue = backend_queue
                self.backend.sc_params = hp
                self.backend.dystart = self.frontend.dystart
                self.backend.iters = iters
                self.backend.set_hyperparams()
                self.backend.rigid_loss = rigid_loss


                

                self.params_gui = gui_utils.ParamsGUI(
                    pipe=self.pipeline_params,
                    background=self.background,
                    gaussians=self.gaussians,
                    q_main2vis=q_main2vis,
                    q_vis2main=q_vis2main,
                )

                backend_process = mp.Process(target=self.backend.run)
                if self.use_gui:
                    gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
                    gui_process.start()
                    time.sleep(5)

                backend_process.start()

                if load_path is None:
                    self.frontend.run()

                backend_queue.put(["sync_backend_final"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()
                    if data[0] == "sync_backend_final" and frontend_queue.empty():
                        mapping_time = data[-1]
                        break

                backend_queue.put(["pause"])

                end.record()
                torch.cuda.synchronize()
                # empty the frontend queue
                N_frames = len(self.frontend.cameras)
                FPS = N_frames / (start.elapsed_time(end) * 0.001)
                Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
                Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
                Log("Mean Mapping Time:", mapping_time, tag="Eval")

                if self.eval_rendering:
                    kf_indices = self.frontend.kf_indices
                    if load_path is None:
                        self.gaussians = self.frontend.gaussians
                    ATE = eval_ate(
                        self.frontend.cameras,
                        self.frontend.kf_indices,
                        self.save_dir,
                        0,
                        final=True,
                        monocular=self.monocular,
                        )

                    rendering_result = eval_rendering(
                        self.frontend.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        kf_indices=kf_indices,
                        iteration="before_opt",
                        save_interval=save_interval,
                        interp_type=self.interp_type,
                    )
                    columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
                    metrics_table = wandb.Table(columns=columns)
                    metrics_table.add_data(
                        "Before",
                        rendering_result["mean_psnr"],
                        rendering_result["mean_ssim"],
                        rendering_result["mean_lpips"],
                        ATE,
                        FPS,
                    )
                    
                    if load_path is None:
                        save_gaussians(self.gaussians, self.save_dir, "final_before_opt", final=False)
                    
                        #save deform before
                        if config["model_params"]["dynamic_model"] == 'mlp':
                            self.gaussians.deform.save_weights(self.save_dir, 80000)
                        elif config["model_params"]["dynamic_model"] == 'offset':
                            self.gaussians.save_offsets(self.save_dir, 80000)
                        
                    
                    # re-used the frontend queue to retrive the gaussians from the backend.
                    while not frontend_queue.empty():
                        frontend_queue.get()
                    
                    backend_queue.put(["color_refinement"])
                    while True:
                        if frontend_queue.empty():
                            time.sleep(0.01)
                            continue
                        data = frontend_queue.get()
                        if data[0] == "sync_backend_final" and frontend_queue.empty():
                            gaussians = data[1]
                            self.gaussians = gaussians
                            mapping_time = data[-1]
                            break

                    rendering_result = eval_rendering(
                        self.frontend.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        kf_indices=kf_indices,
                        iteration="after_opt",
                        save_interval=save_interval,
                        interp_type=self.interp_type,
                    )
                    metrics_table.add_data(
                        "After",
                        rendering_result["mean_psnr"],
                        rendering_result["mean_ssim"],
                        rendering_result["mean_lpips"],
                        ATE,
                        FPS,
                    )
                    wandb.log({"Metrics": metrics_table})

                    if load_path is None:
                        save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
                        #save deform after
                        if config["model_params"]["dynamic_model"] == 'mlp':
                            self.gaussians.deform.save_weights(self.save_dir, 81500)
                        elif config["model_params"]["dynamic_model"] == 'offset':
                            self.gaussians.save_offsets(self.save_dir, 81500)

                backend_queue.put(["stop"])
                backend_process.join()
                Log("Backend stopped and joined the main thread")
                if self.use_gui:
                    q_main2vis.put(gui_utils.GaussianPacket(finish=True))
                    gui_process.join()
                    Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dynamic", action="store_true", default=False)  # 4D dynamic
    parser.add_argument("--yolo", type=int, default=1)  # 4D dynamic
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--inherit_from', type=str, default=None)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--rigid_loss', type=int, default=4.0)
    parser.add_argument("--save_results", type=int, default=1)

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config, inherit_from=args.inherit_from)
    config['yolo'] = bool(args.yolo)
    save_dir = None

    if args.eval:
        Log("Running 4DGS-SLAM in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = args.save_results
        # No GUI supported in eval mode
        config["Results"]["use_gui"] = False 
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = False

        config["Results"]["save_results"] = bool(args.save_results)


    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = config["Dataset"]["dataset_path"].split("/")
    save_dir = os.path.join(
        config["Results"]["save_dir"], path[-2], args.exp_name, path[-1]+ "_" +current_datetime
    )
    tmp = args.config
    tmp = tmp.split(".")[0]
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)
    with open(os.path.join(save_dir, "config.yml"), "w") as file:
        documents = yaml.dump(config, file)
    Log("saving results in " + save_dir)
    run = wandb.init(
        project="4DGS-SLAM",
        name=f"{tmp}_{current_datetime}",
        config=config,
        mode=None if config["Results"]["use_wandb"] else "disabled",
    )
    wandb.define_metric("frame_idx")
    wandb.define_metric("ate*", step_metric="frame_idx")

    save_interval = args.interval
    slam = SLAM(config, 
                save_dir=save_dir, 
                save_interval=save_interval, 
                load_path=args.load_path, 
                iters=args.iters,
                rigid_loss=args.rigid_loss)
    
    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
