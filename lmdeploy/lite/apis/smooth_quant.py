# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from pathlib import Path
import shutil
import json
import fire
import torch
from torch import nn


from lmdeploy.lite.apis.calibrate import (LAYER_TYPE_MAP, NORM_TYPE_MAP,
                                          calibrate)
import lmdeploy
from lmdeploy.lite.apis.calibrate import calibrate, get_model_and_tokenizer

from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            awq_layers, smooth_layers)
from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.models import QLinear, QRMSNorm


def generate_quantize_config(work_dir: str, filename: str = 'quantize_config.json'):
    '''
    为当前生成quantize_config.json配置文件，该文件由vllm读取
    '''
    # 配置内容
    config = {
        "bits": 8,
        "group_size": -1,
        "quant_method": "smooth_quant",
        "linear_only": "true"
    }

    # 文件名称
    config_file_name = osp.join(work_dir, filename)

    # 将配置写入 JSON 文件
    with open(config_file_name, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    print(f"配置文件 {filename} 已成功生成。")
    return 


def smooth_quant(model: str,
                 work_dir: str = './work_dir',
                 calib_dataset: str = 'ptb',
                 calib_samples: int = 128,
                 calib_seqlen: int = 2048,
                 search_scale: bool = False,
                 batch_size: int = 1,
                 w_bits: int = 8,
                 device: str = 'cuda'):

    model_path = model
    work_dir = Path(work_dir)
    if not (work_dir / 'inputs_stats.pth').exists():
        vl_model, model, tokenizer, work_dir = calibrate(model,
                                                     calib_dataset,
                                                     calib_samples,
                                                     calib_seqlen,
                                                     work_dir,
                                                     device,
                                                     w_bits=w_bits,
                                                     w_group_size=-1,
                                                     search_scale=search_scale,
                                                     batch_size=batch_size)
    else:
        vl_model, model, tokenizer = get_model_and_tokenizer(model)

    # calibrate function exports the calibration statistics
    # (inputs, outputs, keys and values) to `work_dir`.
    inp_stats = torch.load(work_dir / 'inputs_stats.pth')
    act_scales = inp_stats['absmax']

    model_type = type(model).__name__
    if model_type not in LAYER_TYPE_MAP or model_type not in NORM_TYPE_MAP:
        raise RuntimeError(
            f'Currently, quantification and calibration of {model_type} are '
            f'not supported. The supported model types are '
            f"{', '.join(LAYER_TYPE_MAP.keys())}.")

    if model_type == 'QWenLMHeadModel':
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError(
                'When using Qwen, you need to `pip install flash-attn` first, '
                'otherwise calibration and quantification will not work '
                'properly.')

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]
    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]

    layers = collect_target_modules(model, layer_type)
    fcs = {}
    for l_name, layer in layers.items():
        name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
        fcs.update(name2fc)

    if search_scale:
        awq_ratios = inp_stats['ratios']
        act_scales = inp_stats['absmean']
        awq_layers(layers, fc2fcs, norm2fcs, act_scales, awq_ratios, -1,
                   device)
    else:
        migration_scales =  smooth_layers(layers, fc2fcs, norm2fcs, act_scales, -1, device)

    # rmsnorms = collect_target_modules(model, norm_type)

    for name, linear in fcs.items():
        linear.to(device)
        
        if name in migration_scales:
            q_linear = QLinear.from_float(linear)
            q_linear.migration_scale.copy_(migration_scales[name].view(1, -1))
        else:
            # else处理类似OPT模型中的fc2和out_proj模块，暂时不对其进行量化迁移，因此scale设置为1
            q_linear = QLinear.from_float(linear)
            input_dim=linear.weight.shape[1]
            q_linear.migration_scale.copy_(torch.ones(1, input_dim))

        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_linear)
        linear.to('cpu')
        
    #TODO 需要注释掉该段落
    # for name, norm in rmsnorms.items():
    #     norm.to(device)
    #     q_norm = QRMSNorm.from_float(norm)
    #     parent_name, _, child_name = name.rpartition('.')
    #     parent = model.get_submodule(parent_name)
    #     setattr(parent, child_name, q_norm)
    #     norm.to('cpu')

    # 因为不需要lmdeploy载入该模型，而仅仅使用lmdeploy的导出，因此不需要这个过程
    # 这样就不需要auto_map的属性
    # if hasattr(model.config, 'auto_map'):
    #     model.config.auto_map.update(AUTO_MAP[type(model).__name__])
    # else:
    #     model.config.auto_map = AUTO_MAP[type(model).__name__]

    if vl_model:
        from .auto_awq import save_vl_model
        save_vl_model(vl_model, model_path, work_dir)
    else:
        model.config.update(
            dict(quantization_config=dict(quant_method='smooth_quant')))
        model.save_pretrained(work_dir,
                              max_shard_size='2GB',
                              safe_serialization=False)
    tokenizer.save_pretrained(work_dir)

    generate_quantize_config(work_dir)
    
    # 因为不需要lmdeploy载入该模型，而仅仅使用lmdeploy的导出，因此不需要这个过程 类似modelling_llama.py
    # pytorch/modeling/modeling_llama.py
    # shutil.copy(MODEL_PATH_MAP[type(model).__name__], work_dir)


if __name__ == '__main__':
    fire.Fire(smooth_quant)
