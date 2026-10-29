# 冻结方案的新时期最终评测

用户最新标准：主对比表仅比较完整c方案与LEAR、RNN、独立Chronos、独立LightGBM；广东/德卢×MAE/RMSE共四项中，至少三项第一，其余最多第二。消融另行完整报告，主对比达标不等于c单独作用已证明。

已有三种子验证中，冻结的h对齐Chronos残差+LightGBM+c(lambda=1)方案已满足此放宽标准：广东MAE第二、其余三项第一。其广东c对打乱消融的微弱优势并未稳定复现，因此继续保留该限制，不宣称c普遍改善。为避免无必要的验证搜索，未启动备选稳健损失实验；不改变已选平方损失与参数。条件c分组结果也完整保存，不将其替换为最终方案。

最终方案固定为此前三种子确认的cas_10：安全目标h对齐历史价/价差与本地特征、Chronos P50加LightGBM残差、c只进损失、lambda1、训练期80/20净化探针300轮、均值偏差方向和日块SE软强度≤.5。新时期只按相同规则重新计算训练期方向/强度和拟合模型，早停仍普通验证RMSE。不得根据新测试成绩换lambda、损失、窗口、种子或重新挑模型。

两个市场各新增第四窗：前80%训练、随后约10%验证、最后约10%测试。边界只读时间列确定，新test首个目标必须严格晚于此前三个有效评测窗口全部目标的最大时间。旧split helper在部分边界多包含一个小时，本脚本显式排除该小时；如不能找到至少192小时的新末段则停止。必须在实际prepare后再次验证无目标时间交叉。不声称整个原始CSV从未被访问；所指新时期是此前未用于有效性能比较和方法选择的目标时段。无任何目标值修改。

先落盘不可变plan.json，再准备新数据。各方法全部使用相同新划分/样本；旧基线结果只读保留，新时期必须新训练对照。LightGBM比较cas_10、普通残差、固定非对称、打乱权重、同h独立树、普通绝对融合，各3种子；独立树剔除上游点/c及点衍生价差。全部下游树训练和验证轮数写入TRAINING_FROZEN.json后才进行最终评分；外部LEAR AIC与RNN64设置也在plan写入前冻结。

新增36份树拟合、6份RNN和2份确定性LEAR拟合（LEAR在三个种子统计中显式复用），Chronos同一冻结模型。最终54条评价记录=2市场×3种子×9方法。统一用float64计算原始单位MAE/RMSE。主表外部排名单独保存MAIN_COMPARISON_RANKS.csv；所有消融/种子/预测保存。主表排名不纳入后续loss搜索。无论是否达标，完整报告此一次新时期结果，不能按测试输赢追调。

```bash
python -u scripts/run_cas_final_period.py --phase check
python -u scripts/run_cas_final_period.py
```

输出`results/cas_final_later_period_20261029`，含边界审计、数据校验和、训练校准与冻结记录、ALL_RUNS.csv、SUMMARY.csv、MAIN_COMPARISON_RANKS.csv和completion.json。RNN六任务可在GPU队列并行，LEAR在CPU，Chronos两个市场并行准备。旧文件与基线不覆盖。
