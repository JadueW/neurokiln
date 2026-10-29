# 在线与闭环扩展点

NeuroKiln 的核心是离线、可复现的评测。在线 / 闭环模块只提供**轻量扩展协议与离线回放**，用于把已经验证过的离线解码器接入逐试次决策流程，不包含实时采集硬件驱动，也不承诺临床或在线性能。

> 当前能力：用留出折的真实预测回放为数据流，逐试次执行解码、置信度阈值与动作决策，并输出可审计事件。实时放大器、LSL / BrainFlow 采集与低延迟调度留给外部适配器。

## 1. 三个协议

`neurokiln.online` 只依赖 NumPy 与引擎数据模型：

| 协议 | 方法 | 作用 |
|---|---|---|
| `StreamSource` | `__iter__() -> Iterator[StreamTrial]` | 按采集顺序逐试次提供数据 |
| `OnlineDecoder` | `predict(trial) -> Prediction` | 只使用当前试次，不读取未来试次 |
| `ClosedLoopPolicy` | `decide(prediction, trial) -> str \| None` | 把预测映射为动作；返回 `None` 表示放弃 |

```python
from neurokiln.online import Prediction, StreamTrial

trial = StreamTrial(
    trial_id="t00042",
    data=epoch,              # (n_channels, n_times)
    timestamp=12.5,          # 秒
    true_label=2,            # 在线时可省略
    subject_id="S01",
    session_id="0",
)
```

`StreamTrial.data` 必须是二维、有限值的 `float` 数组；`Prediction` 会校验概率范围与置信度，未显式给出置信度时取概率最大值。

## 2. 离线回放

`ReplaySource` 按 `Split.test` 的顺序遍历留出试次，`ReplayDecoder` 复用该折已经算出的 `y_pred` / `y_proba`，因此回放不会重新拟合，也不会接触训练折：

```python
from neurokiln.online import ThresholdPolicy, replay_from_split

closed = replay_from_split(
    epochs,
    split,
    split_result,             # GraphPipeline.run_split / Pipeline.run_split 的返回值
    policy=ThresholdPolicy(threshold=0.6, action_template="select:{label}"),
    hop_seconds=1.0,          # 仅生成事件时间戳，不模拟真实延迟
)
print(closed.summary())
# {'n_events': 28, 'n_labeled': 28, 'n_correct': 19, 'accuracy': 0.679,
#  'n_actions': 0, 'n_abstained': 28, 'action_rate': 0.0}
```

`ClosedLoopResult.to_payload()` 会展开逐试次事件：`trial_id`、`timestamp`、真实标签（若有）、预测标签、置信度、动作与决策时间。回放结果只用于检查决策策略与阈值，不等价于在线延迟或硬件性能测试。

## 3. CLI

```bash
# 已安装引擎时
neurokiln online-replay --dataset synthetic --pipeline csp_lda \
  --protocol within_subject --folds 4 --fold 0 --seed 0 \
  --threshold 0.6 --action-template "select:{label}" --hop-seconds 1.0 --json

# 从仓库根目录、未安装引擎时
PYTHONPATH=engine .venv312/Scripts/python.exe -m neurokiln.cli online-replay \
  --dataset synthetic --pipeline raw_centroid --folds 3 --threshold 0.6 --json
```

参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset` | `synthetic` | 数据集 ID |
| `--subjects` | 全部 | 可重复或逗号分隔 |
| `--classes` | 全部 | 类别数 |
| `--options` | 空 | 数据集 JSON 选项 |
| `--pipeline` | `csp_lda` | 内置、`custom:<id>` 或 `plugin:<plugin>/<pipeline>` |
| `--protocol` | `within_subject` | `within_subject` / `cross_session` / `loso` |
| `--folds` / `--fold` / `--seed` | `5` / `0` / `0` | 只回放指定一折 |
| `--threshold` | `0.6` | 置信度低于阈值时放弃动作 |
| `--action-template` | `select:{label}` | 动作模板，支持 `{label}` 与 `{trial}` |
| `--hop-seconds` | `1.0` | 相邻事件的时间间隔 |
| `--json` | 关闭 | 输出机器可读 JSON |

注意：最近质心类基线的 `predict_proba` 常接近 0.5，默认阈值 0.6 会让它们全部放弃动作；这正说明阈值策略需要按解码器概率校准结果设置，而不是照搬默认值。

## 4. 接入实时数据

实时适配器只需实现两个协议，不需要修改引擎：

```python
from neurokiln.online import Prediction, StreamTrial, run_closed_loop


class MyAmplifierSource:
    def __iter__(self):
        while True:
            epoch = read_next_epoch()          # 你的采集 / 分段代码
            yield StreamTrial(
                trial_id=next_id(),
                data=epoch,
                timestamp=time.monotonic(),
                subject_id="S01",
            )


class MyLiveDecoder:
    def predict(self, trial: StreamTrial) -> Prediction:
        proba = self.model.predict_proba(trial.data[None, ...])[0]
        return Prediction(
            label=int(proba.argmax()),
            probabilities=proba,
            classes=tuple(self.model.classes_),
        )


result = run_closed_loop(MyLiveDecoder(), MyAmplifierSource(), ThresholdPolicy(0.7))
```

接入时必须自行保证：

- 解码器只在训练数据上拟合，在线阶段不更新标签或使用未来试次；
- 采集端的滤波、重参考、分段窗口与离线训练一致；
- 丢弃 / 缓冲区溢出 / 时间戳漂移有明确处理策略；
- 动作执行端有自己的急停与人工接管机制。

## 5. 边界与安全

- 该模块**没有**实时性保证、时钟同步、设备驱动、LSL / BrainFlow 集成或网络传输实现。
- 回放使用离线预测，不能用于宣称在线准确率、延迟或闭环增益。
- 在线实验同样需要伦理审批、知情同意与数据访问控制；平台不提供医疗用途保证。
- 若把在线适配器作为插件分发，仍受 `NEUROKILN_PLUGIN_TRUST` 与无沙箱执行的约束，详见[安全模型](/security)。
