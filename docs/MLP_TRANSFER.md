# MLP损失迁移实验

本实验只替换下游回归器，不改变已有Chronos推理、特征构造、训练/验证/测试划分、方向诊断及置信度公式。原LightGBM入口不修改。

默认复现五种配置：`tsfm_point`（无c、L2）、`tsfm_point_C`（有c、L2）、`tsfm_point_fixed_asym_noC`（无c、固定非对称）、`tsfm_point_C_fixed_asym`（有c、固定非对称）、`tsfm_point_C_confidence_asym`（有c、置信度加权非对称）。

核心结论来自最后两项的成对比较。`--suite core`只运行有c的L2、固定非对称和加权非对称三项，各项输入完全相同。不得把不同输入之间的差异全归因于损失。

## 服务器先检查GPU占用

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
ps -eo user,pid,etime,args | less
```

用`nvidia-smi`进程表中的PID在`ps`中查用户名，例如`ps -o user,pid,etime,args -p 12345`。显存低或瞬时利用率为0不代表GPU未被他人预留；集群有调度系统时先申请资源（如Slurm的`squeue`/`salloc`），仅使用分配给自己的卡。容器可能看不到宿主机的用户名；不要终止他人进程。

## 拉取代码

已有仓库有未提交改动时不要直接切分支，优先另建目录：

```bash
git clone --branch mlp-loss-transfer --single-branch https://github.com/ccq040810/CAS-EFP.git CAS-EFP-mlp
cd CAS-EFP-mlp
git rev-parse HEAD
```

已克隆该分支后更新：

```bash
git status --short
git pull --ff-only origin mlp-loss-transfer
```

激活原实验使用的Python环境，不必重装或升级已有CUDA/PyTorch。仓库不包含原始数据、Chronos权重和结果缓存；按原路径放置数据与配置，或软链接原实验目录的`data/`和`models/`。检查：

```bash
python -c "import torch,numpy,pandas,pyarrow,lightgbm,chinese_calendar; print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())"
python -m unittest discover -s tests -p 'test_mlp_transfer.py' -v
```

本地安装依赖时MLP训练只需PyTorch、NumPy、pandas；准备特征还需要原项目的数据/Chronos环境（包括pyarrow、chinese-calendar、chronos-forecasting等）。本脚本不自动安装或更换PyTorch。

## 一次准备特征

必须指向原实验的窗口划分和缓存，避免重新按比例划分。以下绝对路径需替换为服务器上真实位置；不要将本地Windows路径照搬。

```bash
export OLD_SPLITS=/absolute/path/to/results/rolling_adaptive_v3
export OLD_CACHE=/absolute/path/to/results/rolling_adaptive_cache
export CHRONOS2_MODEL_PATH=/absolute/path/to/models/chronos-2

CUDA_VISIBLE_DEVICES=0 python scripts/run_mlp_transfer.py prepare \
  --split-root "$OLD_SPLITS" --cache-root "$OLD_CACHE" \
  --model-path "$CHRONOS2_MODEL_PATH" --device cuda:0 \
  --markets guangdong reale_delu --windows 1 2 3
```

准备阶段复用原缓存；缓存缺失会调用原Chronos流程补全，并写入指定缓存目录。不要同时运行多个prepare去写同一缓存。这里的`0`须替换成获准使用的GPU ID。

每个窗口生成不可变的`features.npz`和`manifest.json`。manifest记录训练方向、特征名、样本规模、数据/配置/缓存哈希、代码版本和时间顺序。如旧目录含预测文件，自动核对测试标签、行顺序、时间及已有Chronos点预测，不一致即报错，不会静默继续。

若需确认准备过程，先只跑`--markets guangdong --windows 1`，正式准备时改用新的`--prepared-root`。已有准备目录不会覆盖；训练时直接复用。

## 多卡运行

小型MLP采用任务并行：每张卡同时跑一个独立的市场×窗口×配置×种子任务；不是用DDP把一个小网络拆到多张卡。

```bash
tmux new -s cas-mlp
python scripts/run_mlp_transfer.py run \
  --gpus 0 1 2 3 \
  --markets guangdong reale_delu --windows 1 2 3 \
  --seeds 42 43 44 --suite original \
  --threads 4 --batch-size 8192 --epochs 300 --patience 30
```

`--gpus`必须改成自己可用的卡。调度器设置了`CUDA_VISIBLE_DEVICES`时，应传已分配卡的GPU UUID（可从该环境变量获取），不要误用宿主机其他卡。脚本对子进程设置可见设备，子进程内部统一使用`cuda:0`。仅CPU测试可用`--gpus cpu`。

默认五配置×两市场×三窗口×三种子=90个独立训练任务；`--suite core`为54个。GPU数增加只增加并发任务数，不保证线性加速；每个进程还会持有自己的CPU特征数组，先确认内存足够。显存不足优先减小`--batch-size`；吞吐低时结合CPU占用调整，不要自动抢占其他卡。

按`Ctrl+B`再按`D`离开tmux，重新连接用`tmux attach -t cas-mlp`。

断点继续需使用完全相同的参数并加`--resume`：仅跳过配置与特征哈希一致且已有`metrics.json`的完成任务，未完成任务从头训练。更换超参数、种子集合、早停规则或suite时使用新`--output-root`。

## 保持原协议的边界

- MLP固定为128→64隐藏层、ReLU、Adam、学习率0.001；默认最多300 epoch、patience 30。结构、学习率不从测试集选择。
- 输入的缺失值中位数填补、标准化及目标标准化仅拟合训练集。这是MLP数值训练所需处理，不新增特征，不改变样本或标签。输出还原到原流水线的目标尺度。
- α=2、λ=1；d由原始训练目标减Chronos点预测的残差中位数确定，一个窗口内固定。
- 加权损失为`mean(0.5*b*(y-pred)^2)`：重点方向b=α[1+λ(1-c)]，另一方向b=1；不会把置信度权重乘到另一方向，也不会除以权重总和。
- 默认`--selection legacy`保持原代码的模型选择：普通配置用验证RMSE；非对称配置先将重点方向误差乘α再计算RMSE（等效平方权重α²），验证不加c权重。该指标与训练损失不完全一致，文稿必须明确。
- 可显式用`--selection rmse --output-root results/mlp_transfer_rmse`统一按验证RMSE早停，但这属于额外协议实验；不能声称与旧LightGBM结果只差回归器。若采用它做主结果，LightGBM也应按相同规则重跑。
- 三种子不挑最优。初始权重和批次顺序在同一窗口、同一输入配置、同一种子下保持一致。
- 本脚本验证的是下游回归器迁移，不验证Chronos分布来源的通用性。

## 监控与结果

```bash
watch -n 2 nvidia-smi
tail -f results/mlp_transfer/logs/guangdong_w1_tsfm_point_C_confidence_asym_seed42.log
find results/mlp_transfer -name metrics.json | wc -l
python scripts/run_mlp_transfer.py summarize
cat results/mlp_transfer/completion.json
```

每个任务独立写日志、模型`model.pt`（含训练集预处理统计）、训练曲线`history.csv`、验证/测试预测和`metrics.json`。`metrics.json`最后写入，作为完成标志。保存的模型含PyTorch状态和NumPy统计，仅加载可信的自产文件。

汇总文件：

- `per_run.csv`：每个窗口/种子/配置的MAE、RMSE、最佳epoch。
- `per_seed.csv`：每个种子先对三窗口等权平均。
- `summary.csv`：再对种子报告均值和标准差；该标准差是种子波动，不是窗口波动。
- `paired_cas_vs_fixed.csv`：同市场、同种子下CAS相对固定非对称的RMSE变化百分比，负值为改善。
- `completion.json`：计划任务、已完成数量及缺失清单。任务缺失时仅导出逐次结果，不输出可能误导的部分平均。

这些指标沿用流水线数据单位（GD通常CNY/MWh、DE-LU通常EUR/MWh），不自动转为论文kWh单位。按原论文相同单位转换后对比；不可将此处原始值与已除以1000的论文表直接相减。

判断迁移时首先比较同一MLP中的`tsfm_point_C_fixed_asym`与`tsfm_point_C_confidence_asym`，报告所有窗口及种子，不能只报告收益最大的时期。不要要求MLP绝对精度必须超过LightGBM。

GPU操作依据：[NVIDIA nvidia-smi文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
