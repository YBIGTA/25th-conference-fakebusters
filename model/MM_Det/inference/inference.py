
import os
import sys
from .modules import inference_frame_maker, LMM_inference

# MM_Det 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import sys
import torch
import os
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn

from options.test_options import TestOption
from utils.trainer import Trainer
from utils.utils import get_logger, get_test_dataset_configs,  set_random_seed
from dataset import get_test_dataloader, get_inf_dataloader
from builder import get_model


# mm_representation_path = '/root/frame_diffusion_detection/MM_Det/inference/mm_representation'
# reconstruction_path = '/root/frame_diffusion_detection/MM_Det/inference/reconstruction'

# _video_dir = '/root/frame_diffusion_detection/MM_Det/inference/quantized.mp4'
# _mother_dir = '/root/frame_diffusion_detection/MM_Det/inference/reconstruction/'
# prefix = _video_dir.split('/')[-1].split('.')[0]
# _image_dir = f'{reconstruction_path}/{prefix}/0_real'


# config = {'data_root': f'{_image_dir}', 
#           'ckpt': '/root/frame_diffusion_detection/MM_Det/weights/MM-Det/current_model.pth',
#           'lmm_ckpt': 'sparklexfantasy/llava-1.5-7b-rfrd', 
#           'lmm_base': None, 
#           'st_ckpt': 'weights/ViT/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth', 
#           'st_pretrained': True, 
#           'model_name': 'MMDet',
#           'expt': 'MMDet_01',
#           'window_size': 10, 
#           'conv_mode': 'llava_v1', 
#           'new_tokens': 64, 
#           'selected_layers': [-1], 
#           'interval': 200, 
#           'load_8bit': False, 
#           'load_4bit': False, 
#           'seed': 42, 
#           'gpus': 1, 
#           'cache_mm': True, 
#           'mm_root': f'{mm_representation_path}/{prefix}/0_real/original/mm_representation.pth', 
#           'debug': False, 
#         #   'cached_data_root': f'{reconstruction_path}', 
#           'output_dir': f'{mm_representation_path}', 
#           'output_fn': 'mm_representation.pth',
#           'mode' : 'inference',
#           'num_workers': 1,
#           'sample_size': -1,
#           'classes': ['inference'],
#           'bs': 1,}




# Example usage


# inference_frame_maker(
#     mother_dir=_mother_dir,
#     video_dir=_video_dir,
# )
# LMM_inference(
#     image_dir = _image_dir,
#     config = config)


def inference(_video_dir, _mother_dir, _image_dir, _mm_representation_path, _reconstruction_path, _config):
    prefix = _video_dir.split('/')[-1].split('.')[0]
    _image_dir = f'{_reconstruction_path}/{prefix}'
    inference_frame_maker(
        mother_dir=_mother_dir,
        video_dir=_video_dir,
    )
    LMM_inference(
        image_dir = _image_dir,
        config = _config)


    # args = TestOption().parse()
    
    config = _config
    logger = get_logger(__name__, config)
    inf_dataset_config = {'inference': {'data_root': f'{config["data_root"]}', 
                                        'dataset_type': 'VideoFolderDatasetForReconsWithFn', 'mode': 'test', 
                                        'selected_cls_labels': [('0_real', 0)], 'sample_method': 'entire'}}
    
    set_random_seed(config['seed'])
    config['st_pretrained'] = False
    config['st_ckpt'] = None   # disable initialization
    

    model = get_model(config)
    model.eval()
    path = None
    if os.path.exists(config['ckpt']):
        logger.info(f'Load checkpoint from {config["ckpt"]}')
        path = config['ckpt']
    elif os.path.exists('expts', config['expt'], 'checkpoints'):
        if os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')):
            logger.info(f'Load best checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')
        elif os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')):
            logger.info(f'Load latest checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')
    if path is None:
        raise ValueError(f'Checkpoint not found: {config["ckpt"]}')
    state_dict = torch.load(path)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v

    video_name = _video_dir.split('/')[-1].split('.')[0]
    mm_path = f'inference/mm_representation/{video_name}/0_real/original/mm_representation.pth'
    config['mm_root'] = mm_path
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=config['cache_mm'])
    
    
    inf_config = deepcopy(config)
    inf_config['datasets'] = inf_dataset_config
    #breakpoint()
    trainer = Trainer(
        config=inf_config, 
        model=model, 
        logger=logger,
    )

    trainer.inference_dataloader = get_inf_dataloader(inf_dataset_config)
    if 'sample_size' in config:    # evaluation on sampled data to save time
        stop_count = config['sample_size']
    else:
        stop_count = -1
    results = trainer.inference_video()
    return results

    # for metric, value in results['metrics'].items():
    #     logger.info(f'{metric}: {value}')
        
# results = inference(_video_dir, _mother_dir, _image_dir, mm_representation_path, reconstruction_path, config)
# print(f'{results}% True')
