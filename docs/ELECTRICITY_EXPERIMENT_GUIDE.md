# 电价实验：本地准备、远程 GPU 运行

## 研究问题和实验顺序

研究任务是广东实时电价和 Real-E 德卢日前电价预测。冻结 Chronos-2 只接收历史目标价格，产生点预测和模型原生有限分位网格；代码在该网格上对分位函数作梯形积分，得到逐预测点的不确定性 `U`，再用训练期 `std(U)` 映射为 `C=exp(-U/std_train(U))`。不把有限网格说成连续模型原生输出，也不把 `num_samples` 当成独立样本数。

LightGBM 的输入包括预测期 `h`、时间日历、预测起点之前的 AR 目标价格特征、Chronos-2 点预测，以及实验开关决定的 `U` 或 `C`。未来实际负荷/发电量不进入特征；目前纳入的物理变量均为日前预测/计划列。Real-E 的 target 为日前电价；广东 target 为实时电价，日前电价与日前系统变量只在有数据时作为未来已知输入。

先运行 `stage1`：AR-LightGBM 强基线、Chronos 点预测、原始 `U`、归一化 `C`，以及 `C` 输入与 `C` 样本加权的区分。看 `confidence_diagnostics_valid.csv`：随着置信度升高，TSFM 原始误差是否下降；树模型误差是否相对下降；低置信度时树是否更偏离 TSFM。它是阶段决策的诊断集（同时用于 early stopping，因此只作筛选性证据）；不要按它调 `alpha`。`confidence_diagnostics_test.csv` 只用于最终、事后报告，不用来决定是否启动 stage2。另看 `train_direction_diagnostics.csv`，它只汇总训练期季度内 TSFM 残差 `y - TSFM` 的方向稳定性。

只有当训练期方向跨季度足够稳定、且 stage1 显示 C 是有用的输入侧可靠性信号时，才运行 `stage2`。stage2 固定 `alpha=2`，`auto` 只根据训练期的中位残差确定惩罚方向；它对比固定非对称损失与 C 调节的方向性样本权重。不要按测试集结果挑方向、挑 alpha 或决定是否报告非对称分支。若方向不稳定，保留并报告对称损失结果，不跑/不主张非对称协同。

`C` 在推理时只依赖 TSFM 输出和训练期拟合的尺度，不依赖真实标签；因此与基于测试残差的 hard-example mining 不同。当前实验把“C 作为树特征”和“C 调节非对称惩罚权重”分开做消融，避免把贡献混成一项。

## 本地数据准备与检查

本地数据/模型不提交到 Git。先在仓库根目录运行：

```bash
python scripts/align_guangdong.py
python scripts/align_reale.py
```

预期输出：

- `data/GD/guangdong_aligned.csv`：`y` 是实时价格；`da_*` 是日前价格/物理预报；`rt_*` 不进入实验特征。
- `data/REALE/DE-LU_aligned.csv`：目标为 `y_Day-ahead Price [EUR/MWh]`；实验配置仅使用日前发电计划、日前负荷预测和确定日历变量，不使用任何 `Actual` 列。

`align_reale.py` 使用本地 Real-E 包中的 `Real-E/OLD`（对应数据发布包的 Original 版）；对齐脚本保留 actual 列用于研究数据归档，但模型配置不会选入。DST 附近时间戳、重复/缺失小时和输出行数应在正式跑大实验前检查。当前 aligned 文件必须是连续小时数据；有断点时不要将行号 lag 当作小时 lag，需先修复时间轴。

当前本地 DE-LU 对齐结果总计 83,256 行，但安全配置中的日前计划/预测和目标同时非空的行约 49,456 行，存在缺失区间。因此正式结果必须记录有效窗口数；不能把原始总行数直接当成样本数。若某个时间切分有效窗口过少，应缩短实验时段或只报告该数据集的可用连续窗口。

Chronos-2 权重不进仓库。确保 `CHRONOS2_MODEL_PATH` 指向含 `config.json` 的本地模型目录；当前开发机路径例子是 `models/chronos-2/chronos-2`。

## 本地代码验证

要求 Python 3.12+。用项目环境安装后先跑语法与小测试：

```bash
uv sync
python -m compileall -q run_pipeline.py exp data_provider ts_models scripts
python -m unittest discover -s tests -v
```

无 NVIDIA GPU 时，将 `DEVICE=cpu` 并将样本量缩小做管线冒烟；Chronos-2 全量推理建议远端 GPU。完整运行时不要打开 `--tsfm_force_current_rerun`，除非确实需要重建分布缓存。

## 远端服务器初次准备

```bash
git clone https://github.com/ccq040810/CAS-EFP.git
cd CAS-EFP
uv sync
```

将两份原始数据集放到仓库预期目录：`Dataset/Guangdong electricity market data/` 和 `Real-E/OLD/`；将 Chronos-2 权重放到服务器可访问路径。上传大文件可用 `rsync`/`scp`，不要将数据或权重 commit 到 Git。若本地和远程数据目录不同，先调整脚本的配置路径或同步到约定目录。`uv sync` 安装 torch 后如服务器 CUDA 环境与锁文件 wheel 不兼容，应按服务器 CUDA 版本安装匹配的 PyTorch，再运行 `uv sync --no-install-project` 或依项目环境锁定方式完成其余依赖。

## 远端实验命令

```bash
python scripts/align_guangdong.py
python scripts/align_reale.py
export CHRONOS2_MODEL_PATH=/path/to/chronos-2
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda
bash scripts/electricity/run_two_dataset_ablation.sh stage1
```

先确认两个数据集都成功生成 stage1 结果和诊断文件，再人工按上述规则决定是否执行：

```bash
bash scripts/electricity/run_two_dataset_ablation.sh stage2
```

可用 `SEQ_LEN=168 PRED_LEN=24 BATCH_SIZE=16 N_JOBS=8` 调整资源占用。缓存位于每个数据集的 `results/electricity/<dataset>/tsfm_cache_qmad_v1/`，跨消融复用；每组 LightGBM 结果保存在独立子目录。

## 计划内消融矩阵

| 阶段 | 对照 | 所回答的问题 |
|---|---|---|
| 1 | AR-LightGBM | 不依赖 TSFM 的强基线 |
| 1 | TSFM point（无 U/C）+ AR | 加 TSFM 点预测是否有增益 |
| 1 | point + 原始 U + AR | 分布尺度作为特征是否有效 |
| 1 | point + C + AR | 训练尺度归一化是否优于原始尺度 |
| 1 | point + C + confidence-weighted MSE + AR | C 输入特征与样本加权的区别 |
| 2 | point + C + 固定非对称损失 + AR | 固定方向损失是否改善 |
| 2 | point + C + C 调节非对称损失 + AR | C 是否进一步调节方向性惩罚 |

所有模型用同一时间切分、同一目标和相同测试集。重点按数据集单独报告 MAE/RMSE 及高价/尖峰子集结果；不把两个市场拼在一起训练或合并总体指标。
