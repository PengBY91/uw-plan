---
title: W2-07 EuRoC MH_01_easy / MH_02_easy 公共基线（A0 OpenVINS）
created: 2026-10-12
type: report
status: execution-record
---

# W2-07 EuRoC MH_01_easy / MH_02_easy 公共基线

**结论：两条序列都按 W1 的 1× 回放协议复现成功，MH_01 两次和 MH_02 有效的两次运行逐项一致；整段 SE(3) 对齐 APE 为 0.073 m（MH_01）和 0.070 m（MH_02）。** MH_02 的第二次运行（r02）在回放时丢了 543 个 IMU 样本和 45 对图像，随后发散到 APE 21 m。按协议保留这次失败，又补跑了 r03，结果与 r01 一致。四次有效运行都没通过 W1 回归门槛中的"位姿覆盖 ≥ 0.95"，原因是 MH 序列开头有 39–45 s 的静止和抬升，OpenVINS 在这段时间没有输出；这个门槛是为 V1_01 定的，这里只如实记录未通过，不调整门槛。所有结果只按本项目协议评分，不宣称追平 OpenVINS 论文。

## 数据获取与审计

| 环节 | 实际情况 |
| --- | --- |
| 来源 | ETH Research Collection `machine_hall.zip`（12,683,729,426 字节，MD5 363f5c25…，见 `locks/eth-files.json`）。没有下载整个包：`scripts/data/download_euroc.py --list-entries` 用 HTTP Range 读 ZIP64 中央目录（用同样方法重建的 vicon_room1 条目与历史的 `locks/archive-entries.json` 逐字节一致），然后只取 `MH_01_easy`、`MH_02_easy` 的 `.bag` 和 `.zip` 四个成员 |
| 校验 | 每个成员都核对了 Content-Range，解压后 CRC32 与中央目录一致，并记录了本地 SHA256（`manifests/euroc_mh01.yaml`、`euroc_mh02.yaml`）。没有校验整包 MD5，因为没有下载整包 |
| 限流 | 服务器按请求次数返回 HTTP 429，且不带 Retry-After；16–128 MB 分块下载基本卡死。下载器改为每个成员只发一个流式 Range 请求，中断时从本地断点续传，退避时间为分钟级 |
| 转换与审计 | `scripts/data/prepare_euroc.py --sequence`：rosbags 0.11.6 转为 ROS2 Jazzy MCAP；`audit_euroc.py` 逐条比对 ROS1 与 ROS2 的全部消息（排除 Header.seq），并与 ASL 文件夹核对图像像素抽样、时间戳（≤128 ns）和 IMU 数值（≤1e-6）。MH_01：IMU 36820，双目各 3682，Leica 3099；ASL 7384 个条目通过 CRC。MH_02：IMU 30400，双目各 3040；ASL 6100 个条目通过 CRC。两条序列的审计都通过 |
| 真值 | 每条序列只用自己的真值，不混用。OpenVINS 自带的 `ov_data/euroc_mav/MH_0x_easy.csv` 与 ETH 原始的 `state_groundtruth_estimate0/data.csv` SHA256 相同（OpenVINS 只修正了 V1_01），所以下表中"修正"和"原始"两套参考给出的 APE 相同 |

## 运行与结果

配置直接沿用 `configs/euroc_v101`：MH 与 V1 是同一套传感器和标定，配置中包括在线时间偏移和在线相机标定。命令和 W1 replay-fix 相同（1×、reliable IMU、history depth 1000、Cyclone DDS），只是加了 `--dataset-audit` 指向本序列的审计。代码提交为 505c800，OpenVINS 为 b64db17（另一会话的诊断版构建，只增加了日志输出）。评估采用 W1 协议：`evaluate_openvins.py` 做整段 SE(3) 对齐、不拟合尺度、关联容差 10 ms，RPE 取实际时间戳对；风险模式另由 `uwbench eval-traj` 计算，按前 10 s 窗口做偏航加平移对齐，窗口内不评分，容差为暂定的 0.15 m / 3°。

| 运行 | 输入（IMU；双目） | 首个输出延迟 | 相机槽覆盖 | APE RMSE / 最大（m） | RPE 1 s / 5 s（m） | 风险模式最大误差 x / y / z（m），航向（°） | 越限时长 x / y / z / 航向（s） | 回归门槛 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| euroc_mh01_a0_r01 | 36820/36820；3682/3682 | 44.9 s | 0.756 | 0.073 / 0.187 | 0.029 / 0.086 | 0.104 / 0.256 / 0.257，4.26 | 0 / 32.8 / 26.9 / 27.4 | 仅位姿覆盖未过 |
| euroc_mh01_a0_r02 | 36820/36820；3682/3682 | 44.9 s | 0.756 | 0.073 / 0.187 | 0.029 / 0.086 | 同 r01 | 同 r01 | 仅位姿覆盖未过 |
| euroc_mh02_a0_r01 | 30400/30400；3040/3040 | 39.1 s | 0.743 | 0.070 / 0.151 | 0.021 / 0.054 | 0.174 / 0.197 / 0.174，0.97 | 11.5 / 3.7 / 5.4 / 0 | 仅位姿覆盖未过 |
| euroc_mh02_a0_r02 | **29857/30400；2995/3040** | 39.1 s | 0.735 | **21.379 / 132.837** | 4.284 / 17.252 | 61.7 / 121.7 / 80.7，2.20 | 67.7 / 84.8 / 74.9 / 0 | 输入完整性、位姿覆盖、无断档、APE、RPE 均未过 |
| euroc_mh02_a0_r03 | 30400/30400；3040/3040 | 39.1 s | 0.743 | 0.070 / 0.151 | 0.021 / 0.054 | 同 r01 | 同 r01 | 仅位姿覆盖未过 |

估计与参考的匹配率为 0.990–0.992；未匹配的估计都落在参考有效时间之外（MH_01 有 22 个），不是关联失败。回放墙钟 155–187 s，与序列时长一致。

**MH_02 r02 的失败。** 输入审计显示，OpenVINS 回调实际收到的数据比 bag 少了 543 个 IMU 样本和 45 对图像，输出出现 7 段超过 75 ms 的断档，随后滤波发散。这次运行持锁执行，日志中没有其他需要锁的任务与之重叠；同一时段只有本会话的故障批处理做了 3 s 的单线程 bag 读取和评分。另一会话可能在 Windows 侧生成仿真数据，会占用宿主机 CPU，但无法确认。原因未定位。r03 在同一构建和同一 bag 上零缺样，结果与 r01 逐项相同。这次失败说明 1× 回放对宿主机负载仍然敏感，W1 replay-fix 记录的"同一构建可复跑"只在负载干净时成立。

**风险模式越限。** MH_01 在 y、z 和航向上各有约 27–33 s 越限，MH_02 在 x 上越限 11.5 s。整段 SE(3) 对齐下误差小得多，因为风险模式只用前 10 s 窗口对齐，而 MH 的前 10 s 正好处在刚完成初始化后的晃动阶段，窗口拟合的偏航误差会在之后几十米的航程上放大。按交接要求，对齐窗口按 W1 协议执行，不为好看另选。

## 与 OpenVINS 论文协议的差异

本报告的数值不能和 OpenVINS 论文的 EuRoC 表格直接比较：

- 论文的 ATE 通常采用位置加偏航（4 自由度）对齐，并对多次运行取平均；这里报的是整段 SE(3) 对齐，单次运行逐条列出。
- 论文以离线最快速度或作者自己的设置运行；这里是 ROS2 下 1× 实时回放，算法只看 cam0、cam1 和 imu0。
- 配置沿用本项目的 `configs/euroc_v101`，没有为 MH 单独调参。

因此只说明"在本项目协议下复现成功、复跑一致"，不声称追平或优于论文数值。

## 局限

- 每条序列只有两次有效运行；MH_02 的一次失败原因没有定位。
- 回归门槛来自 V1_01（期望消息数改为由本序列审计给出），位姿覆盖门槛不适合开头静止时间长的序列，这里未修改门槛。
- 风险模式容差仍是暂定值，待 `configs/acceptance_w2.yaml` 冻结。

## 复现命令

```bash
S=<staging>; cp locks/eth-files.json locks/eth-item.json $S/
.venv-eval/bin/python scripts/data/download_euroc.py --stage $S --sequence MH_01_easy --list-entries
.venv-eval/bin/python scripts/data/download_euroc.py --stage $S --sequence MH_01_easy
.venv-eval/bin/python scripts/data/prepare_euroc.py --lab "$PWD" --stage $S --sequence MH_01_easy --report reports/w2/euroc-mh --manifest manifests/euroc_mh01.yaml
RUN=euroc_mh01_a0_r01; AUD=$PWD/reports/w2/euroc-mh/mh_01_easy-data-audit/audit.json; BAG=$PWD/data/public/euroc/MH_01_easy_ros2
bash scripts/env/ros_cyclone_run.sh /usr/bin/python3 scripts/openvins/run_openvins.py --lab "$PWD" --bag $BAG --run-id $RUN --rate 1.0 --imu-history-depth 1000 --imu-reliability reliable --dataset-audit $AUD
.venv-eval/bin/python scripts/openvins/audit_input_trace.py --bag $BAG --run $PWD/runs/$RUN
.venv-eval/bin/python scripts/openvins/evaluate_openvins.py --run $PWD/runs/$RUN --audit $AUD --gt-original $PWD/data/public/euroc/MH_01_easy_asl/mav0/state_groundtruth_estimate0/data.csv --gt-corrected $PWD/ros_ws/src/open_vins/ov_data/euroc_mav/MH_01_easy.csv --sequence MH_01_easy
.venv-eval/bin/python scripts/openvins/check_replay_acceptance.py --run $PWD/runs/$RUN --dataset-audit $AUD
PYTHONPATH=team/uwbench .venv-eval/bin/python -m uwbench eval-traj --manifest manifests/euroc_mh01_run.yaml --estimate runs/$RUN/estimate_bag --out runs/$RUN/eval-traj
```

`evaluate_openvins.py` 的 `--run` 必须给绝对路径，因为它在 `evaluation/` 子目录里调用 `evo_traj`。第一轮批处理传的是相对路径，四次运行的评估都失败了；失败留下的目录已改名为 `evaluation.failed-relative-path/` 保留，之后用绝对路径重评。
