---
title: W1 WSL 环境盘点与代码仓库准备记录
created: 2026-10-10
updated: 2026-10-10
type: plan
status: execution-record
tags: [acoustic-optic-fusion, slam, 3d-reconstruction, rov]
---

# W1 WSL 环境盘点与代码仓库准备记录

本次执行 [[plans/acoustic-optic-month1-execution-checklist-2026-10|首月清单]]的环境盘点和仓库准备，工作区实际建立在 WSL 的 `/home/steve/uw-ao-lab`。采用已有 Ubuntu 24.04 与 ROS 2 Jazzy，评测使用独立 Python 3.12 venv。原 [[plans/acoustic-optic-month1-environment-runbook-2026-10|运行手册]]中的 Humble 命令仍是参考配置，本机使用下文 Jazzy 入口。

环境与仓库准备不等于完成 W1：公共数据运行、D0 采集、标定审计、完整 B 几何链与融合后端审计仍需执行。缺少或尚未核验的输入集中在 [[plans/acoustic-optic-w1-missing-inputs-2026-10|W1 数据 SDK 与配置缺口清单]]。日期按本轮工作日记为 2026-10-10；机器证据另存 UTC 时间。

## 实测环境与选择

| 项目 | 实际检查结果 | 本次处理与适用边界 |
| --- | --- | --- |
| WSL | Ubuntu，WSL2；用户 steve | 使用原生 Linux 工作目录；旧工程只读核查 |
| OS 与内核 | Ubuntu 24.04.4 LTS；6.6.87.2-microsoft-standard-WSL2 | 保留，不在 24.04 安装 Humble |
| CPU | Intel Core Ultra 9 185H；WSL 可见 22 个逻辑 CPU | OpenVINS 按顺序构建包，编译并行数设 2 |
| RAM | 23 GiB；swap 16 GiB | 以实际编译结果判断容量，不把交换空间当物理内存 |
| 磁盘 | WSL ext4 可用约 847 GiB；宿主 C 盘可用约 466 GiB | VHD 标称空间不代表物理可用空间；大数据下载按宿主余量控制 |
| GPU | RTX 4070 Laptop GPU；8188 MiB；驱动 596.58；存在 `/dev/dxg` | `nvidia-smi` 成功；未做 PyTorch CUDA 推理验收，W1 A0 不依赖 GPU |
| ROS | `/opt/ros/jazzy`；加载后 ros2 与 rclpy 可用 | cv_bridge、image_transport、message_filters、RViz2、rosbag2 的 MCAP/SQLite3 插件已存在 |
| 编译工具 | Git 2.43.0；CMake 3.28.3；g++ 13.3.0；Make 4.3 | 使用系统工具链 |
| 系统 Python | 3.12.3 | ROS 使用系统 Python；eval 使用该版本创建的独立 venv |
| 数学与视觉库 | Eigen 3.4.0；OpenCV 4.6.0；Boost 1.83 | 与系统 ROS 保持一致 |
| Ceres | 初查系统未安装；现补齐 Ubuntu 包 2.2.0+dfsg-4.1ubuntu2 | 新装仅 libceres-dev、libceres4t64；下载文件与本机 APT 索引 SHA256 比对通过 |
| OctoMap | 初查无系统开发包；现编译源码 1.10.0 | 安装到 `~/uw-ao-lab/local`，不覆盖系统目录 |
| 已有 Conda | uw_slam_build 中 Ceres 2.2、Eigen 5.0.1、Python 3.14 系列 | 不混入 ROS 编译；保留原环境和 HoloOcean 环境 |
| 设备节点 | 检查未发现 video、ttyUSB、ttyACM 节点 | 只说明此次节点检查结果；网络声呐的连通性未测 |
| 网络 | WSL 的 GitHub、PyPI、Ubuntu HTTPS 探测超时；Windows 提权执行的 Git HTTPS 可用 | 经 Windows 临时目录取得源码与 Linux wheel 后导入 WSL；WSL 直连问题仍待修复 |

选择依据：[ROS Jazzy 官方安装说明源码](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Installation/Ubuntu-Install-Debs.rst)将 Ubuntu Noble 24.04 列为目标；[OpenVINS 安装说明](https://docs.openvins.com/gs-installing.html)给出 ROS2 colcon 构建入口。二者不替代本机的实际编译与运行验收。

## 工作区和版本

```text
/home/steve/uw-ao-lab/
  ros_ws/src/open_vins/          A0 上游源码
  vendor/{octomap,SVIn,SonarSweep,SonarSweep_dataset}/
  .venv-eval/                    evo 与 rosbags 等离线工具
  local/                        本地 OctoMap 头文件和库
  team/                         OctoMap 冒烟源程序和可执行文件
  scripts/                      环境检查、ROS 隔离入口、构建入口
  locks/                        仓库 SHA、Python 依赖、wheel SHA256
  cache/wheels/                 Linux Python 3.12 离线安装包
  manifests/                    数据位置登记和 V1_01 待补模板
  configs/platform.pending.yaml 平台待核验字段，不能作为算法参数运行
  reports/w1/                   命令、编译与烟测证据
  data/{public,pool}/            新数据预留目录，目前未填充
```

| 仓库 | 锁定提交 | 来源及准备结果 |
| --- | --- | --- |
| [OpenVINS](https://github.com/rpng/open_vins) | `69488123ed9362dd44b6f28e7f4680abbff1442b` | Windows 获取官方仓库后导入 WSL，origin 恢复为官方 URL |
| [OctoMap](https://github.com/OctoMap/octomap) | `b5f9a9598f734609974d1086a785c54d25ec8763` | 官方源码；Release 编译、本地安装和链接烟测通过 |
| [SVIn](https://github.com/AutonomousFieldRoboticsLab/SVIn) | `fcda5466e08d955c5207fdc7e05f6eb57bf0fc7e` | 从团队已有 checkout 克隆已提交内容；未构建，供 W1-09 审计 |
| [SonarSweep 算法](https://github.com/LIAS-CUHKSZ/SonarSweep) | `b189606109c2dd6184c1aeefc54910b12a1b904f` | 官方算法源码已就位；未安装网络推理栈、下载模型或运行测试 |
| [SonarSweep 数据转换](https://github.com/LingpengChen/SonarSweep_dataset) | `d4db20dffeba41b857eb5e604a78578ef816a8c7` | 从本机同名数据目录识别出的另一仓库，单独保存，不冒充算法仓库 |

这里锁定的是本次取得或复用的提交，不宣称均为最新发行版。仓库全量来源、子模块状态和本地补丁状态见 `locks/repositories.json`；重新准备时用相同提交与补丁，不直接跟随上游 HEAD。没有新建远程团队仓库。

已有 `uw_slam`、`oasis_iros_codex`、`underwater_ood_scheduling` 等工程存在本地修改；本次未重置、提交或覆盖这些工程。已有 Oculus 驱动路径 `~/ros2_ws/src/sonar_oculus` 是指向旧工程的符号链接，只确认源码存在，未确认适配采购声呐。

## 验收结果

| 检查 | 结果 | 可追溯证据 |
| --- | --- | --- |
| 系统、硬件、ROS、Python | 通过盘点 | `reports/w1/environment-probes.json` |
| 四个计划指定仓库和一个数据转换仓库 | 已准备并记录 SHA | `locks/repositories.json` |
| eval 环境 | pip check、六个模块导入、evo_ape 和 rosbags-convert 帮助均通过 | 环境探测 JSON；`locks/eval-requirements.txt` |
| OctoMap 编译与链接 | 通过 | `octomap-configure.log`、`octomap-build.log`、`octomap-install.log`、`octomap-smoke.log` |
| OctoMap 三态 | 已知占据、已知空闲和未观测位置检查通过，分辨率 0.1 m | `team/octomap_smoke.cpp`；这不是 W1-08 的完整双目与遮挡几何验收 |
| OpenVINS 四包编译 | 兼容补丁后全部通过；保留两次失败与最终成功日志 | `openvins-build-01.log`、`openvins-build-02.log`、`openvins-build-03.log` |
| OpenVINS 入口 | 列出 run_subscribe_msckf 等四个可执行文件；launch 参数查询通过 | `executables.log`、`launch_args.log` |
| 重复构建与补丁 | 同版本增量构建通过；补丁反向检查通过；未做独立空目录完整重建 | `incremental_rebuild.log`、`patch_reverse_check.log`、`acceptance-checks.json` |
| SVIn 融合能力 | 仅完成入口检查，未通过融合复现 | 锁定分支 README 明确 ROS2 下 sonar/depth 模式未就绪；不能因名称含声呐就认定可用 |
| GPU 神经网络、实机链路、公共 A0 回放 | 未验收 | 分别需要推理环境、设备输入和完整公共数据 |

评测主要版本为 evo 1.37.1、rosbags 0.11.6、NumPy 2.2.6、SciPy 1.16.3、pandas 2.3.2、PyYAML 6.0.3；完整依赖冻结和离线 wheel 哈希已保存。SonarSweep 原 requirements 同时包含两种 OpenCV wheel，且固定 NumPy 1.24.4，不应直接安装进本机 Python 3.12 ROS/eval 环境。

OpenVINS 的本机版本为“上游 SHA + `locks/openvins-jazzy.patch`”。补丁只修改 `ROS2Visualizer.h`、`ROSVisualizerHelper.h` 中 image_transport、cv_bridge 和 tf2_geometry_msgs 的四处头文件包含，优先 `.hpp`、回退 `.h`，不改变估计器逻辑。首次失败是旧头文件路径不存在；最终四包构建退出码为 0，之后增量构建再次退出 0。初始日志中的编译告警保留，编译通过不替代轨迹数值验收。

W1-01 已完成；W1-02 已完成仓库锁定、依赖准备、编译和入口检查，因尚未在独立空构建目录复建而保留未勾选。W1-03 至 W1-09 未整项验收，不能将本次工作记为 G0 通过。

## 可直接执行的检查入口

以下在 WSL Bash 执行。ROS 包装脚本清理继承的 Conda 环境，只加载 Jazzy、本工作区和本地库；它服务于离线构建与 CLI 检查，实机联网、显示和 DDS 参数需后续明确配置。

```bash
export AO_LAB="$HOME/uw-ao-lab"
python3 "$AO_LAB/scripts/audit_w1.py" --lab "$AO_LAB"
bash "$AO_LAB/scripts/ros-run.sh" ros2 pkg list
bash "$AO_LAB/scripts/ros-run.sh" ros2 pkg executables ov_msckf
bash "$AO_LAB/scripts/ros-run.sh" ros2 launch ov_msckf subscribe.launch.py --show-args
"$AO_LAB/.venv-eval/bin/python" -m pip check
"$AO_LAB/.venv-eval/bin/evo_ape" --help
"$AO_LAB/team/octomap_smoke"
bash "$AO_LAB/scripts/build-openvins.sh"
```

离线重建 eval 环境时另用新目录，保留现有环境：

```bash
python3 -m venv "$AO_LAB/.venv-eval-rebuild"
"$AO_LAB/.venv-eval-rebuild/bin/python" -m pip install --no-index \
  --find-links "$AO_LAB/cache/wheels" -r "$AO_LAB/locks/eval-requirements.txt"
"$AO_LAB/.venv-eval-rebuild/bin/python" -m pip check
```

上述重建命令是后续操作入口，不能据此认定已经独立重建。检查脚本和精选机器证据共 34 个文件，归档在 `sources/projects/2026-10-10-w1-environment/`；同时保留 WSL 原件。现有 Git 白名单默认忽略其中 Python、JSON、日志和补丁，因此这批原始证据目前是本机归档，不会随普通 wiki 提交自动分发；本页的环境、提交号、补丁说明和验收结论可随 Markdown 文档提交。大体积源码、wheel、构建产物和数据保留 WSL，不进入知识库 Git。
