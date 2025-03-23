import os
import sys
import traceback
from copy import deepcopy
import torch

from .modules import inference_frame_maker, LMM_inference
from options.test_options import TestOption
from utils.trainer import Trainer
from utils.utils import get_logger, get_test_dataset_configs, set_random_seed
from dataset import get_test_dataloader, get_inf_dataloader
from builder import get_model

def inference(_video_dir, _mother_dir, _image_dir, _mm_representation_path, _reconstruction_path, _config):
    try:
        prefix = _video_dir.split('/')[-1].split('.')[0]
        _image_dir = f'{_reconstruction_path}/{prefix}'
        print(f"🔍 Inference 시작 - video: {prefix}")
        
        print("📦 Step 1: Frame 생성 중...")
        inference_frame_maker(
            mother_dir=_mother_dir,
            video_dir=_video_dir,
        )

        print("🧠 Step 2: LMM Inference 중...")
        LMM_inference(
            image_dir=_image_dir,
            config=_config
        )

        config = _config
        logger = get_logger(__name__, config)
        logger.info("⚙️ Step 3: 모델 설정 시작...")

        inf_dataset_config = {
            'inference': {
                'data_root': f'{config["data_root"]}',
                'dataset_type': 'VideoFolderDatasetForReconsWithFn',
                'mode': 'test',
                'selected_cls_labels': [('0_real', 0)],
                'sample_method': 'entire'
            }
        }

        set_random_seed(config['seed'])
        config['st_pretrained'] = False
        config['st_ckpt'] = None

        model = get_model(config)
        model.eval()

        logger.info(f"📁 Checkpoint 경로 확인 중: {config['ckpt']}")
        path = None
        if os.path.exists(config['ckpt']):
            logger.info(f"✅ Checkpoint 로드: {config['ckpt']}")
            path = config['ckpt']
        elif os.path.exists('expts', config['expt'], 'checkpoints'):
            best_ckpt = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')
            latest_ckpt = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')
            if os.path.exists(best_ckpt):
                logger.info(f"✅ Best Checkpoint 로드: {best_ckpt}")
                path = best_ckpt
            elif os.path.exists(latest_ckpt):
                logger.info(f"✅ Latest Checkpoint 로드: {latest_ckpt}")
                path = latest_ckpt

        if path is None:
            logger.error(f"❌ Checkpoint를 찾을 수 없습니다: {config['ckpt']}")
            raise ValueError(f"Checkpoint not found: {config['ckpt']}")

        logger.info(f"📥 모델 파라미터 로딩 중...")
        state_dict = torch.load(path, map_location="cpu")
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=config['cache_mm'])

        video_name = _video_dir.split('/')[-1].split('.')[0]
        mm_path = f'inference/mm_representation/{video_name}/0_real/original/mm_representation.pth'
        config['mm_root'] = mm_path

        logger.info(f"📊 Step 4: Inference 시작")
        inf_config = deepcopy(config)
        inf_config['datasets'] = inf_dataset_config

        trainer = Trainer(config=inf_config, model=model, logger=logger)
        trainer.inference_dataloader = get_inf_dataloader(inf_dataset_config)

        stop_count = config.get('sample_size', -1)
        results = trainer.inference_video()

        logger.info(f"✅ Inference 완료: {results}")
        return results

    except Exception as e:
        print("🔥 [ERROR] inference() 함수 실행 중 예외 발생:")
        traceback.print_exc()
        return {
            "message": f"Exception during inference: {str(e)}",
            "status": "error"
        }
