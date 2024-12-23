import os
import io
import yaml
import argparse
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.ppg_c import PPG_C
from utils.ppg_g import PPG_G
from utils.roi import ROIProcessor


def process_video(video_path: str, config: dict) -> dict:
    ROI = ROIProcessor(video_path=video_path, config=config)
    masked_frames = ROI.detect_with_draw()
    transformed_frames, _ = ROI.detect_with_map(crop=False)
    R_means, L_means, M_means, fps = ROI.detect_with_calculate(crop=False)

    results = {
        "frame_per_second": int(fps),
        "masked_frames": masked_frames,
        "transformed_frames": transformed_frames,
        "R_means": R_means.T,
        "L_means": L_means.T,
        "M_means": M_means.T
    }
    return results

def compile_video(video_path: str, masked_frames: np.ndarray, transformed_frames: np.ndarray, fps: int):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    masked_path = os.path.join("misc/ppg/output", video_name+"_mask.mp4")
    transformed_path = os.path.join("misc/ppg/output", video_name+"_transformed.mp4")
    skvideo.io.vwrite(masked_path, masked_frames, inputdict={'-r': str(fps)})
    skvideo.io.vwrite(transformed_path, transformed_frames, inputdict={'-r': str(fps)})

def create_graph(video_path: str, R_means: np.ndarray, L_means: np.ndarray, M_means: np.ndarray, fps: int):
    G_R = PPG_G.from_RGB(R_means, fps).compute_signal()
    G_L = PPG_G.from_RGB(L_means, fps).compute_signal()
    G_M = PPG_G.from_RGB(M_means, fps).compute_signal()
    C_R = PPG_C.from_RGB(R_means, fps).compute_signal()
    C_L = PPG_C.from_RGB(L_means, fps).compute_signal()
    C_M = PPG_C.from_RGB(M_means, fps).compute_signal()
    signals = np.vstack([G_R, G_L, G_M, C_R, C_L, C_M])
    min = np.min(signals)
    max = np.max(signals)

    def create_frame(signals, min, max):
        fig, ax = plt.subplots(figsize=(6, 4))        
        ax.plot(signals[0,:], label='G_R', color='red')
        ax.plot(signals[1,:], label='G_L', color='blue')
        ax.plot(signals[2,:], label='G_M', color='green')
        ax.plot(signals[3,:], label='C_R', color='orange')
        ax.plot(signals[4,:], label='C_L', color='purple')
        ax.plot(signals[5,:], label='C_M', color='brown')
        ax.set_ylim(min, max)
        ax.set_title("PPG Signals")
        ax.legend(loc='upper right')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)
    
    frames = [create_frame(signals[:,:i], min, max) for i in range(signals.shape[1])]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    graph_path = os.path.join("misc/ppg/output", video_name+"_graph.mp4")
    skvideo.io.vwrite(graph_path, frames, inputdict={'-r': str(fps)})


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Process video and generate output.")
    parser.add_argument('--video_path', '-v', type=str, help="Path to the input video file")
    parser.add_argument('--config_path', '-c', type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()

    video_path = args.video_path
    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    result = process_video(video_path, config)
    compile_video(video_path, result["masked_frames"], result["transformed_frames"], result["frame_per_second"])
    create_graph(video_path, result["R_means"], result["L_means"], result["M_means"], result["frame_per_second"])