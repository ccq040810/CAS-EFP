"""
Chronos2 模型适配器
"""
import os
import warnings
import torch
import numpy as np
from typing import Union, Optional, Dict, Any, List
from ..base import BaseTimeSeriesModel
import math


class Chronos2Adapter(BaseTimeSeriesModel):
    """
    Chronos2 模型适配器
    
    支持的模型路径示例：
    - amazon/chronos-2
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, model_type: str = 'chronos2'):
        """
        初始化 Chronos2 适配器
        
        Args:
            model_path: Huggingface 模型ID或本地路径，默认为 'amazon/chronos-2'
            device: 设备，如 'cuda' 或 'cpu'
            model_type: 模型类型标识，默认为 'chronos2'
        """
        self.model_type = model_type
        if model_path is None:
            model_path = 'amazon/chronos-2'
        super().__init__('chronos2', model_path, device)
    
    def load_model(self):
        """
        加载 Chronos2 模型
        
        从指定的模型路径加载 Chronos2Model，并配置设备映射和数据类型。
        模型会自动从本地文件加载（local_files_only=True）。
        
        Raises:
            ImportError: 如果缺少必要的依赖包
            RuntimeError: 如果模型加载失败
        """
        try:
            from chronos.chronos2 import Chronos2Model
            from chronos import Chronos2Pipeline
            from transformers import AutoConfig

            if self.model_path is None:
                raise ValueError("Sundial model requires model_path to be specified")
            
            
            model_path = self.model_path
            if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
                raise FileNotFoundError(f"Chronos2 local path missing config.json: {model_path}")
            if os.path.isdir(model_path):
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                assert hasattr(config, "chronos_config"), "Not a Chronos config file"
                self.model = Chronos2Model.from_pretrained(
                    model_path,
                    config=config,
                    local_files_only=True,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    device_map=self.device,
                    dtype=torch.float32
                )
            else:
                self.model = Chronos2Model.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    device_map=self.device,
                    dtype=torch.float32
                )
            
            # self.model.eval()
            self._is_loaded = True
            print(f"Loaded Chronos2 ({self.model_type}) model from {self.model_path}")
        except ImportError:
            raise ImportError(
                "Chronos2 requires the 'chronos2' package. "
                "Install it with: pip install chronos-forecasting"
                "Notice: Chronos2 requires huggingface transformers >= 4.56.2"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos2 model: {e}")
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        context_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        future_covariates: Optional[Union[torch.Tensor, np.ndarray]] = None,
        future_covariates_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_samples: Optional[int] = 1,
        quantile_levels: Optional[List[float]] =None,
        future_target: Optional[Union[torch.Tensor, np.ndarray]] = None,
        future_target_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Chronos2 进行预测
        
        支持多变量时间序列预测，可以处理协变量（covariates）和掩码（mask）。
        内部会将输入转换为 Chronos2 模型所需的格式，并自动处理 batch 和 variate 维度。
        
        Args:
            context: 输入序列，形状为 [batch_size, n_variates, seq_len]
                    支持 torch.Tensor 或 np.ndarray，会自动转换为 torch.Tensor
            forecast_horizon: 预测的时间步长度
            context_mask: 可选的上下文掩码，用于标记缺失值
            future_covariates: 可选的未来协变量，形状为 [batch_size, forecast_horizon, n_covariates]
                               支持 torch.Tensor 或 np.ndarray
            future_covariates_mask: 可选的未来协变量掩码，用于标记缺失的协变量值
            num_samples: 采样次数，会被自动设置为 quantile_levels 的长度
            quantile_levels: 分位数水平列表，如果为 None 则使用模型配置中的默认值
            **kwargs: 传递给模型 forward 方法的其他参数
        
        Returns:
            dict: 包含以下键的预测结果字典
                - 'forecast': torch.Tensor，形状为 [B, C, forecast_horizon, num_quantiles]
                            原生分位数值（非独立样本），最后一维对应 quantile_levels
                - 'quantile_levels': List[float]，原生分位水平列表（升序，如 0.01..0.99）
                - 'mean': torch.Tensor，形状为 [B, C, forecast_horizon]
                        中位数点预测 Q(0.5)（而非对分位数值求平均）
                - 'point_forecast': 同 'mean'
                - 'uncertainty': torch.Tensor，形状为 [B, C, forecast_horizon]
                        MAD 积分 U = (1/(τ_hi-τ_lo)) ∫ |Q(τ)-Q(0.5)| dτ
                - 'metadata': dict，包含模型元数据：
                    - 'model': 模型名称
                    - 'num_samples': 采样次数
                    - 'forecast_horizon': 预测长度
                    - 'num_output_patches': 输出 patch 数量
                    - 'output_patch_size': 输出 patch 大小
        
        Note:
            - 当前实现将多变量输入展平为 [batch_size * n_variates, seq_len] 进行单变量推理
            - group_ids 用于标识同一 batch 中的不同变量样本
            - 如果提供了 future_covariates，会自动处理其维度转换
        """
        self._ensure_model_loaded()
        quantile_levels = self.model.chronos_config.quantiles
        if num_samples != len(quantile_levels):
            warnings.warn(f"Chronos2 is sampled by quantile_levels: {quantile_levels}, setting num_samples to {len(quantile_levels)}")
            num_samples = len(quantile_levels)
        
        
        # [qiuyz] NOTE: This is still a Uni-variate inference, similar data preprocessing to amazon/chronos-2 is needed for Multi-variate and Covariate Inference.
        B, C, L = context.shape

        try:
            assert type(context) == torch.Tensor
        except AssertionError:
            context = torch.from_numpy(context).to(self.device)

        context = self._normalize_input_dim_2D(context) # [batch_size * n_variates, seq_len]

        ps = self.model.chronos_config.output_patch_size
        num_output_patches = math.ceil(forecast_horizon / ps)


        # * group_ids: same id for every variate sample in batch ([0,0,0,1,1,1,2,2,2,...])
        group_ids = torch.repeat_interleave(torch.arange(B), C).to(self.device)

        if future_covariates is not None:
            try:
                assert type(future_covariates) == torch.Tensor
            except AssertionError:
                future_covariates = torch.from_numpy(future_covariates).to(self.device)
            future_covariates = self._normalize_input_dim_2D(future_covariates)
            if future_covariates_mask is not None:
                try:
                    assert type(future_covariates_mask) == torch.Tensor
                except AssertionError:
                    future_covariates_mask = torch.from_numpy(future_covariates_mask).to(self.device)
                future_covariates_mask = self._normalize_input_dim_2D(future_covariates_mask)

        
        with torch.no_grad():
            outputs = self.model.forward(
                context = context,
                context_mask = context_mask,
                future_covariates = future_covariates,
                future_covariates_mask = future_covariates_mask,
                group_ids = group_ids,
                num_output_patches = num_output_patches,
                future_target = None,
                future_target_mask = None,
                **kwargs
                
            )
        outputs_tensor = outputs.quantile_preds[:, :, :forecast_horizon]

        # 最后一维是「原生分位节点」，不是独立随机样本。
        # quantile_values: [B, C, H, Q]，Q = len(原生分位水平)，quantile_levels 严格递增。
        quantile_values = outputs_tensor.reshape(B, C, -1, forecast_horizon).permute(0, 1, 3, 2)

        levels = [float(x) for x in quantile_levels]
        if len(levels) < 2 or any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError(f"Chronos-2 quantile levels must be strictly increasing: {levels}")
        if not any(abs(level - 0.5) < 1e-8 for level in levels):
            raise ValueError(f"Chronos-2 native quantile grid must contain tau=0.5: {levels}")
        levels_t = torch.tensor(levels, device=quantile_values.device, dtype=quantile_values.dtype)

        # 点预测 = 中位数 Q(0.5)（取 τ=0.5 最近的节点）
        median_idx = next(i for i, level in enumerate(levels) if abs(level - 0.5) < 1e-8)
        point_forecast = quantile_values[:, :, :, median_idx].contiguous()  # [B, C, H]

        # 离散度 U^{[.01,.99]} = (1/(τ_hi-τ_lo)) ∫ |Q(τ) - Q(0.5)| dτ（梯形积分，围绕中位数的 MAD）
        abs_dev = (quantile_values - point_forecast.unsqueeze(-1)).abs()  # [B, C, H, Q]
        d_tau = levels_t[1:] - levels_t[:-1]                              # [Q-1]
        integral = ((abs_dev[..., :-1] + abs_dev[..., 1:]) * 0.5 * d_tau).sum(dim=-1)  # [B, C, H]
        uncertainty = (integral / (levels_t[-1] - levels_t[0])).to(quantile_values.dtype)

        return {
            'forecast': quantile_values,       # 原生分位数值 [B, C, H, Q]（非样本）
            'quantile_levels': levels,         # 原生水平列表（升序）
            'mean': point_forecast,            # 中位数点预测 Q(0.5)
            'point_forecast': point_forecast,
            'uncertainty': uncertainty,        # MAD 积分 U
            'metadata': {
                'model': 'chronos2',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon,
                'num_output_patches': num_output_patches,
                'output_patch_size': self.model.chronos_config.output_patch_size,
                'quantile_levels': levels,
            }
        }
