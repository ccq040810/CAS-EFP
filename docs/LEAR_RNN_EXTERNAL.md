# 仅新增LEAR适配版与RNN基线

为适配小时起报，两者共同加入起报小时的sin/cos编码；LEAR末7列仍为星期哑变量。

不重新训练任何现有模型。使用现有v2准备文件的目标值、逐样本时间和相同划分，重新从原始价格及安全日前协变量构建输入。逐行核验标签；不读取TSFM预测，不使用其缓存中的历史。

LEAR适配版：价格D−1/2/3/7的24小时向量，已知外生变量D/D−1/D−7向量，7个星期哑变量；训练侧median/MAD与asinh变换；每个预测步独立LassoLarsIC(AIC)选正则系数，再Lasso拟合。参照epftoolbox的LEAR定义，独立实现：https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/models/_lear.py 。为匹配已有实验，采用每小时起报、每个扩展窗口拟合一次，而非官方午夜起报、逐日重新校准，因此明确标记lear_adapted，不能称为官方基准完整复现。种子不改变其确定性估计，三种子重复便于与现有汇总形式对齐。

RNN：真正的单层64单元tanh循环网络，编码起报前168小时价格；连接同一已知外生变量向量与星期哑变量，经64单元ReLU头输出24步。训练侧缩放，Adam=.001，batch256，最多150轮，普通验证RMSE早停耐心20，梯度范数上限5。模型不是原MLP改名，也不冒称epftoolbox DNN。

两个市场、三个窗口、三个种子、两个模型，共36次。输入历史按时间戳索引；价格缺失仅因果前向填充，无法填充则报错；协变量缺失用训练中位数处理。未来协变量仅使用原安全配置的日前预测/计划字段。新指标为原单位，广东元/MWh，德卢欧元/MWh。

```bash
conda activate efp
cd ~/CAS/CAS-EFP-transfer
python -u scripts/run_lear_rnn.py
```

默认结果目录results/lear_rnn_external_20261029，含ALL_RUNS.csv、SUMMARY.csv、completion.json及逐样本预测。所有既有结果只读。新基线使用正确历史，不可与未来历史故障回放数据计算正式相对提升。
