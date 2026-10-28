# 旧协议模型替换诊断

此实验仅用于重放旧实现，包含已知或待审计的未来历史输入，不是有效预测性能验证。输出以DIAGNOSTIC命名，不能混入v2实验表。

计划：3上游×2下游×2市场×3窗口×3种子×9配置＝972次训练。九配置为原七配置、仅损失使用c而不输入c、完全不使用上游点与c的纯回归器。另外计算各上游在相同测试行上的单独预测误差。纯回归器按上游分别运行，重复结果不代表额外模型家族。

Chronos直接复用reproduction_audit_20261028/old_cache_prepared中的六份旧特征，保持原生分位数及旧缓存原值；这六份特征已重现旧稿。替换上游使用原适配器共同Q10—Q90概率网格，在预测历史选取中执行原整数查找np.searchsorted(index.view('int64'),anchor.value)。原生日期单位不修改，不进行因果填充；Pandas环境不同可能使该旧逻辑选择不同历史或跳过含缺失的历史，因此逐起报点保存HISTORY_AUDIT，不声称它们是严格受控的泛化证据。新模型本来没有旧稿结果，不能声称其概率适配器也已存在于旧稿。

旧树设置：feature_fraction=.9、bagging_fraction=.8、bagging_freq=5、min_gain_to_split=.08、learning_rate=.05、63叶、叶节点最少40样本、2000轮上限、耐心100。非对称早停使用旧规则。MLP沿用128/64隐藏层、学习率.001、batch8192、最多300轮、耐心30，采用同一旧非对称早停定义。α=2、λ=1；无测试集调参。

```bash
conda activate efp
cd ~/CAS/CAS-EFP-transfer
git pull --ff-only origin mlp-loss-transfer
nohup python -u scripts/run_cas_legacy_diagnostic.py > results/legacy_protocol_diagnostic_launcher.log 2>&1 < /dev/null &
```

输出目录results/legacy_protocol_diagnostic_20261028。DIAGNOSTIC_RESULTS.md为可用组汇总，completion.json为完成与缺失清单；DIAGNOSTIC_ALL_RUNS.csv保留逐次训练，DIAGNOSTIC_UPSTREAM_ONLY.csv保存单独上游指标。配置失败如实保存，不自动更改旧设置以取得完整数值。只有窗口齐全的种子参与三窗口平均，种子数列公开实际覆盖范围。

进程锁防重复启动，相同脚本配置可断点续跑；改脚本或配置需另选包含diagnostic的新输出目录。旧时间错误只在诊断进程的局部包装中重放，不改生产历史定位代码。
