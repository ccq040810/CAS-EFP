# CAS上下游模型替换实验

固定原广东和德国—卢森堡数据、训练/验证/测试窗口、168小时历史、24小时预测及特征构造，分别替换上游概率模型和下游回归器。该实验回答CAS的增量收益能否跨模型保持，不预设任何替代模型必然受益。

## 实验矩阵

| 上游 | 下游 | 概率来源 |
| --- | --- | --- |
| Chronos-2 | LightGBM、MLP | 原生分位数 |
| Moirai-2 | LightGBM、MLP | 原生分位数，使用checkpoint中的quantile_levels |
| Sundial | LightGBM、MLP | 100条真实生成轨迹的经验分位数 |

中位数是预测分布的统计量，与模型家族无关。普通单点神经网络的输出不自动等于中位数，也不能据此生成置信度；因此不接入只返回点预测的上游。下游MLP直接输出最终价格，CAS非对称平方损失不要求它预测中位数。

不同家族统一到Q10、Q20、…、Q90九个节点，P50作为点预测，按相同梯形积分计算围绕P50的绝对离散度，再由原特征流程用训练集离散度标准差计算c。原生分位数先进行单调重排，再在原生覆盖范围内插值，不外推尾部；生成轨迹先计算经验分位数，绝不把分位节点当作独立随机样本。

共同网格的Chronos-2结果需要重跑。它与论文原生0.01—0.99网格属于不同概率接口协议，不能把二者的差异全部归因于模型替换。旧Chronos-2入口及其缓存保持可用；新增cas_前缀的模型使用独立缓存、协议身份文件和准备目录。

## 配置与运行

在服务器上使用已有实验环境，准备原始数据、权重和原三窗口data_split.json。不要重新按比例分割。先复制并编辑配置：

```bash
cp configs/cas_transfer.example.json configs/cas_transfer.local.json
```

修改split_root、模型路径、获准使用的GPU。路径相对于仓库根目录，建议原实验资源使用绝对路径。代码固定使用原DATASETS映射：广东为data/GD/guangdong_aligned.csv，德国为data/REALE/DE-LU_aligned.csv；列配置沿用原仓库。数据不上传到GitHub。

```bash
python scripts/run_cas_transfer_matrix.py prepare --config configs/cas_transfer.local.json --dry-run
python scripts/run_cas_transfer_matrix.py prepare --config configs/cas_transfer.local.json
python scripts/run_cas_transfer_matrix.py verify --config configs/cas_transfer.local.json
python scripts/run_cas_transfer_matrix.py run --config configs/cas_transfer.local.json
python scripts/run_cas_transfer_matrix.py summarize --config configs/cas_transfer.local.json
```

prepare按模型依次运行，每个模型只推理一次并保存特征，两个下游复用同一份数组。run运行前强制核对上游之间的标签、测试预测起点—目标时间—预测步、时间序列和非TSFM特征完全相同；概率特征缺失时直接报错。训练方向d由各上游当前窗口的训练残差重新确定，不使用测试标签选方向。

prepare不覆盖已有目录。中断后需重建未完成模型时，修改配置仅包含未完成的上游，并为其指定新的准备根目录；训练可用完全相同配置加--resume跳过完成任务。变更模型权重路径、数据、批大小或协议会触发缓存身份检查，应使用新的cache_root。相同路径内替换权重文件也应手动换缓存目录。

默认selection=rmse，两类下游所有配置都按普通验证RMSE早停。可选legacy复现旧非对称验证口径，但必须单独设置output_root；主交叉实验不要混用两个选择口径。α=2、λ=1。MLP默认128→64、ReLU、Adam，仅用训练集拟合缺失填补和输入/目标标准化。LightGBM使用学习率0.05、63叶、叶节点最少40样本、最多2000轮、耐心100轮。

## 机制对照

suite=original运行原有五配置；core运行同输入的普通平方、固定非对称、CAS三配置；mechanism在五配置基础上增加以下两项：

| 配置 | 修改内容 | 判断用途 |
| --- | --- | --- |
| mean_weight_asym | 损失中的w_i全部改为训练集mean(w) | 检查是否仅由平均惩罚增大带来收益 |
| shuffled_weight_asym | 按种子打乱训练样本的w_i | 检查样本与置信度的对应关系是否有效 |

两项对照均保留真实c输入，只改变损失权重；重点方向以外的系数始终为1。均值匹配仅匹配训练集候选权重的均值，不保证每一轮当前重点残差子集的有效权重均值相同。它也不替代验证集调优固定α的实验。

每个模型组合运行全部窗口和种子，不从测试集选择最优模型、种子或方向。默认3上游×2下游×7配置×2市场×3窗口×3种子，共756次训练。可先在配置中设置单市场、单窗口、单种子、suite=core，检查环境与流程，再换新的output_root进行完整实验。LightGBM在CPU运行，MLP使用指定GPU；不同上游所需依赖可能需要独立环境，可在各环境分别prepare，再共享已准备的数组。

## 结果判断

结果按上游/下游分目录，包含每次训练预测、模型、训练曲线、指标和特征指纹。summary.csv先在每个种子内对窗口等权平均，再汇总种子均值及标准差；paired_by_window.csv保留每个窗口、每个种子的CAS与各对照差值，负百分比表示CAS误差较低。任务未完成时不输出部分平均。

优先看同组合、同窗口、同种子、同输入下CAS相对固定非对称的差异，再结合均值权重与打乱对照解释。MLP绝对误差高于LightGBM不直接意味着机制不适配；若CAS在MLP内也持续退化，或无法优于打乱权重，则应收窄泛化主张并分析原因。不同上游训练方向可能变化，这是原方向诊断规则的结果，需一并报告。

## 验证与依赖

```bash
python -m unittest discover -s tests -p 'test_cas_transfer.py' -v
python -m unittest discover -s tests -p 'test_mlp_transfer.py' -v
```

测试覆盖分位数/样本语义、中位数与积分、拒绝点输出和尾部外推、各适配器输出契约、方向损失梯度与LightGBM一致、训练集预处理、模型训练、任务调度、断点配置校验及跨上游样本一致性。适配器契约使用模拟模型输出；不替代真实权重的服务器验证。测试通过不能据此声称跨模型实验已取得收益。

复用原Chronos环境；Moirai-2需支持该模型的uni2ts及其依赖，Sundial需其官方支持的transformers环境。脚本不自动升级依赖或下载数据。官方接口依据：

- [Moirai-2预测输出与分位水平](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/model/moirai2/forecast.py)
- [Sundial样本生成接口](https://github.com/thuml/Sundial/blob/main/README.md)

本地开发目录尚缺完整广东/德国数据和三套权重，当前仅进行代码与合成数据验证，正式MAE/RMSE需要在原实验服务器运行。
