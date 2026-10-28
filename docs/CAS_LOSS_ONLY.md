# 只将置信度用于损失系数：18次补充实验

新增 `tsfm_point_confidence_asym_noC`：从上游分布计算c，只在训练损失重点方向使用α[1+λ(1−c)]；回归器不读取c。预测CSV保留c供审计，不代表模型输入包含c。

本轮仅运行Chronos-2＋LightGBM：2市场×3窗口×3种子，共18次。复用修复后的v2特征，以及原v2实验中36次无c输入的普通基线、固定非对称对照。保持α=2、λ=1、训练残差自动方向及普通验证RMSE早停，避免同时改变多项设置。旧论文缓存不混入本实验。

```bash
conda activate efp
cd ~/CAS/CAS-EFP-transfer
git pull --ff-only origin mlp-loss-transfer
python scripts/run_cas_loss_only.py --check-only
mkdir -p results
nohup python -u scripts/run_cas_loss_only.py > results/cas_loss_only_launcher.log 2>&1 < /dev/null &
```

使用CPU，默认3个并行任务、每任务4线程，无需下载模型或重新进行GPU推理。重复运行相同命令会核对配置并跳过完成任务；输出目录持有进程锁，拒绝重复启动。恢复时不要改特征、种子、早停或超参数。

```bash
tail -f results/cas_loss_only_launcher.log
cat results/cas_loss_only_20261028/completion.json
cat results/cas_loss_only_20261028/RESULTS.md
```

结果目录 `results/cas_loss_only_20261028`：

- `ALL_RUNS.csv`：54条记录，包含18次新增和36次复用结果。
- `PER_SEED.csv`：每个种子先等权平均三个窗口。
- `FINAL_RESULTS.csv`：两个市场、三个方案的种子均值与样本标准差。
- `PAIRED_BY_WINDOW.csv`：逐窗口逐种子与两个对照的36次配对比较，百分比负值为改善。
- `RESULTS.md`、`completion.json`：最终表格和完成状态；只有全部成功才输出最终表格，失败请查看logs。

默认原单位为广东元/MWh、德卢欧元/MWh；除1000转换成各自币种/kWh。输出分离，不改写旧756次结果。此轮不是新的独立测试集，结果应如实报告，不保证收益；只覆盖原架构，不构成完整跨模型验证。
