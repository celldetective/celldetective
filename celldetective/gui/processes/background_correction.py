from multiprocessing import Process
import time
import os
from pathlib import PurePath, Path
import numpy as np
from tifffile import imwrite
from celldetective.preprocessing import correct_background_model, auto_load_number_of_frames, _get_img_num_per_channel, _extract_channel_indices_from_config
from celldetective.utils import config_section_to_dict, extract_experiment_channels

class BackgroundCorrectionProcess(Process):
    
    def __init__(self, queue=None, process_args=None):
        
        super().__init__()
        
        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)
        
        self.sum_done = 0
        self.t0 = time.time()

    def run(self):

        print('Start background correction process...')
        
        # Load config to get movie length for progress estimation
        self.config = PurePath(self.exp_dir, Path("config.ini"))
        self.len_movie = float(config_section_to_dict(self.config, "MovieSettings")["len_movie"])
        self.nbr_channels = len(extract_experiment_channels(self.exp_dir)[0])
        channel_indices = _extract_channel_indices_from_config(self.config, [self.target_channel])
        self.img_num_channels = _get_img_num_per_channel(channel_indices, self.len_movie, self.nbr_channels)
        self.total_frames = self.len_movie * len(self.position_option) * len(self.well_option)

        self.count = 0
        self.t0 = time.time()

        export = getattr(self, 'export', False)
        return_stacks = getattr(self, 'return_stacks', True)
        movie_prefix = getattr(self, 'movie_prefix', None)
        export_prefix = getattr(self, 'export_prefix', 'Corrected')
        
        def progress_callback():
            self.count += 1
            if self.count > self.total_frames:
               self.count = self.total_frames # prevent overshoot

            self.sum_done = (self.count / self.total_frames) * 100
            if self.sum_done > 100:
                self.sum_done = 100
            
            elapsed_time = time.time() - self.t0
            if self.count > 0:
                avg_time_per_frame = elapsed_time / self.count
                remaining_frames = self.total_frames - self.count
                remaining_time = remaining_frames * avg_time_per_frame
            else:
                remaining_time = 0
                
            self.queue.put([self.sum_done, remaining_time])

        try:
            corrected_stacks = correct_background_model(
                               self.exp_dir,
                               well_option=self.well_option,
                               position_option=self.position_option,
                               target_channel=self.target_channel,
                               model=self.model,
                               threshold_on_std=self.threshold_on_std,
                               operation=self.operation,
                               clip=self.clip,
                               export=export,
                               return_stacks=return_stacks,
                               activation_protocol=self.activation_protocol,
                               show_progress_per_well=False,
                               show_progress_per_pos=False,
                               movie_prefix=movie_prefix,
                               export_prefix=export_prefix,
                               progress_callback=progress_callback,
                               downsample=getattr(self, 'downsample', 10)
                               )

            if return_stacks and corrected_stacks and len(corrected_stacks) > 0:
                 # Save to temp file for the main process to pick up
                 # We assume single position for preview
                 temp_path = os.path.join(self.exp_dir, "temp_corrected_stack.tif")
                 try:
                    imwrite(temp_path, corrected_stacks[0])
                    print(f"Saved temp stack to {temp_path}")
                 except Exception as temp_e:
                    print(f"Failed to save temp stack: {temp_e}")
            
            self.sum_done = 100
            self.queue.put([self.sum_done, 0])

        except Exception as e:
            print(f"Error in background correction process: {e}")
            self.queue.put("error")
            return

        self.queue.put("finished")
        self.queue.close()

    def end_process(self):
        self.terminate()
        self.queue.put("finished")
