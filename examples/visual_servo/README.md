# 第六周：相机标定、手眼标定与视觉伺服

统一入口是 `experiment.py`，网页使用同一份相机图像、内参与机器人模型。默认启动仿真，实物实验需要填写设备参数并主动启用。所有命令从仓库根目录运行。

## 软件启动

已配置的本机环境可以直接运行：

```powershell
.\examples\visual_servo\start.ps1
```

打开 http://127.0.0.1:8767/。点击“开始仿真 / 相机观察”，观察蓝色四角点向绿色目标收敛；点击“遮挡 / 恢复标记”检查目标丢失时输出为零。空格键暂停。

新环境安装所需依赖：

```powershell
uv sync --locked --extra visual-servo --extra feetech --extra test
```

独立检查、仿真与测试：

```powershell
uv run --no-sync python -X utf8 examples/visual_servo/experiment.py preflight
uv run --no-sync python -X utf8 examples/visual_servo/experiment.py verify
uv run --no-sync python -X utf8 examples/visual_servo/task3.py simulate
uv run --no-sync python -X utf8 -m pytest -c examples/visual_servo/pytest.ini examples/visual_servo/tests -q
```

`verify` 保存测试报告、16 张合成棋盘图（12 组训练、4 组验证）、手眼恢复结果、ArUco 闭环视频和逐帧数据。全部位于 `outputs/verification/时间戳/`；其中任务 2/3 结果均为合成数据，报告明确标记未完成实物实验。

## 已完成的任务 1

`results/task1/` 保存 2026-10-12 的真实 USB 相机结果、20 张标定图和 3 张独立验证图：

| 项目           | 结果                            |
| -------------- | ------------------------------- |
| 相机           | GENERAL WEBCAM，原生 1280×720   |
| 棋盘           | 8×6 内角点，方格边长 25 mm      |
| 整体重投影 RMS | 0.828025 px                     |
| 独立验证误差   | 0.512240、0.745967、0.743372 px |

第 17 张训练图的单图误差为 2.897 px，其他 19 张均不超过 0.842 px。任务 2 使用默认 2 px 图像质量门限，因此不会接收这张图用于配对。相机驱动没有确认关闭自动对焦，请保持相同镜头、焦距与分辨率；重新对焦或更换相机后应重新标定。

标定文件内的照片路径已改为相对 `results/task1` 的路径，矩阵、误差和图像像素保持原值；来源记录见 `provenance.json`。这些任务 1 照片没有机械臂同步位姿，不能直接充当真实手眼配对。

## 设备配置

编辑 `configs/robot_model.json`，填写与实物对应的 URDF 路径、基座与相机安装末端帧、活动关节名称和舵机 ID。URDF 路径及 `motor_calibration_path` 相对仓库根目录。当前五关节列表、COM3、末端帧名称只是待核对模板，不能代替实际设备信息。若相机安装使夹爪关节影响相机位姿，必须把对应关节加入 URDF 链和 ID 列表。

默认 `portable_urdf` 后端用 NumPy/SciPy 计算 FK 和有界数值 IK，无需 `placo`。支持 fixed/revolute/continuous 关节链；连续关节也必须指定有限安全范围。不支持的 mimic/prismatic/floating 关节会明确报错。`lerobot_placo` 可选后端调用原仓库的 `RobotKinematics`，需要另装 `kinematics` extra 并满足其运行依赖。

SCS215 的观测是 0～1，不能当作度或弧度：

```text
q_deg = q_at_0_deg + normalized × (q_at_1_deg - q_at_0_deg)
q_rad = q_deg × π / 180
```

每个关节的两个角度端点必须在自己的 URDF 坐标约定下实测；反向关节允许 `q_at_1_deg < q_at_0_deg`。还要填写 `joint_limits_deg` 和自己的舵机原始范围标定 JSON，不能使用旧控制网页作者的标定数据。范围将与 URDF 限位取交集。`motor_calibration_path` 使用原仓库的逐关节字段：`id`、`drive_mode`、`homing_offset`、`range_min`、`range_max`、可选 `range_segments`。

本实验的 SCS215 通信使用 Feetech SDK 和本仓库控制表，运行在独立串口进程；不用加载 PyTorch。只写位置模式的 SRAM 指令，不自动改 ID、波特率、EEPROM 或校准范围。校准弧的映射与 fork 一致，但越界反馈及跨编码器间断区运动会被拒绝。舵机原始目标按最近编码器格点取整，微小建议累积后发送，并按实际相机位移节流；没有达到可发送格点的周期会记录为未发送。默认关节单步上限为 0.006 rad（约 0.344°），仍需按实测角度映射检查是否高于一个编码器格点。

编辑 `configs/experiment.yaml` 的 `marker`：真实 ArUco 字典、ID、**黑色外边框**的实测边长（米），并设置 `confirmed: true`。空值时只使用 4×4/ID 0/60 mm 的仿真示例；这个尺寸不能视为实物测量值。修改配置后重启网页服务。

## 任务 2：真实手眼标定

相机必须刚性安装在机械臂末端。固定棋盘与机械臂基座，贴出棋盘的同一个物理内角点原点。在采集过程中保持它不动。相机、机械臂各由一个进程占用；先正常退出旧的相机采集程序或串口控制程序，再运行：

```powershell
.\examples\visual_servo\start.ps1 -Source camera -Camera 2 -AllowHardware
```

相机编号在重插设备后可能改变，Windows 会核对设备名称。原生尺寸必须为 1280×720。打开网页后连接并读取关节，不会自动启用力矩。如果需要徒手调整姿态，先支撑机械臂，勾选准备项并点击“释放关节，手动采集”；该操作会关闭活动关节力矩。切换到棋盘视图，确认红色原点对应同一个物理角点；只有检测顺序相反时才勾选反转。

每次调整姿态、停稳，再保存训练配对。采集会真实读取拍照前后关节状态，检查变化不超过 0.5°、FK 平移不超过 1 mm，并保存原生照片、角点叠图、时间戳、FK、PnP 和配置哈希。完整棋盘必须可见，默认单图重投影 RMS ≤2 px。建议 15～20 组训练，至少 12 组；绕至少两个不同轴改变姿态（建议有明显的倾斜角度），另留至少 3 组独立验证。

测量照片中棋盘原点在基座坐标系中的 X/Y/Z（米），输入网页并求解。输出保存到 `results/task2/hand_eye.yaml`、JSON 和报告。旋转激励不足、重复照片、重复训练姿态、分辨率变化、关节标定变化或坐标系混用都会阻止保存/求解。

矩阵约定：`T_parent_from_child` 把子坐标系坐标转换到父坐标系。

```text
输入：B_T_E（FK）、C_T_M（棋盘 PnP）
输出：E_T_C = T_end_from_camera
固定棋盘检查：B_T_M = B_T_E × E_T_C × C_T_M
```

求解使用 OpenCV PARK 方法。验证既检查固定棋盘的一致性，也与独立实测原点比较。课程要求每轴偏差 ≤5 cm；实物伺服额外要求至少 3 组独立验证、验证原点误差范数 ≤5 cm、训练原点一致性 RMS ≤5 cm，并且模型及内参哈希一致。报告中的 `validated_for_servo` 是软件门限状态，硬件实验还需核对相机安装方向、尺寸和低速运动方向。

离线导入已有配对也受支持：

```powershell
uv run --no-sync python -X utf8 examples/visual_servo/task2.py init
uv run --no-sync python -X utf8 examples/visual_servo/task2.py fk --model configs/robot_model.json --observation data/task2/observation.json
uv run --no-sync python -X utf8 examples/visual_servo/task2.py add --image data/task2/photo.png --robot-pose data/task2/robot_pose.json
uv run --no-sync python -X utf8 examples/visual_servo/task2.py status
uv run --no-sync python -X utf8 examples/visual_servo/task2.py solve --board-origin-m 0.3 0.1 0.2
```

最后一行的坐标只是命令格式示例，须替换为自己的实测值。普通 `capture --robot-pose ...` 只能检查文件前后是否变化；有同步机器人反馈的真实配对请使用统一网页。不要对已存在的配对数据集再次 `init`。

## 任务 3：四角点 IBVS

默认选择课程进阶方案：检测指定且唯一的 ArUco，保留镜头畸变进行 PnP，再将四角点去畸变并转为归一化图像坐标，用 8×6 图像雅可比计算相机速度。目标是中央四角点，也可在摆好目标姿态后点击“将当前四角点设为目标”。默认停止条件为四角点 RMS ≤2 px。

实物使用 `L × J_camera` 直接求活动关节速度，支持少于六个相机位姿自由度的机械臂；相机偏置包含在 FK 数值雅可比中。不会假设五关节机械臂能到达任意六自由度姿态。默认建议平移速度 ≤0.01 m/s、角速度 ≤0.1 rad/s；位置式舵机的实际瞬时速度和到达效果需在实物低速测试中确认。

`controller.mode: rule` 可切换课程的区域规则：目标在左/右/上/下侧 40% 区域时请求相机相应方向移动 1 cm，中央则前进 1 cm。软件求解该相机目标对应的末端 IK，再拆为限幅关节步；不可达目标会停止，不会当作成功执行。

相机单独观察：

```powershell
.\examples\visual_servo\start.ps1 -Source camera -Camera 2
```

如果暂时保留旧的 8766 相机服务，可用 `-Source broker` 读取其 960×540 缩略 JPEG，明确缩放内参及角点。该来源无法证明原始帧新鲜度，因此不允许真实手眼配对或实物运动。

实物配置与真实手眼结果齐全后，以 `-Source camera -AllowHardware` 启动，先观察和核对方向，再勾选实验准备并点击“启用实物伺服”。启用时才设置当前位置目标和力矩；真实目标丢失、重复 ID、深度越界、PnP 不可靠、过期图像/关节反馈、限位、不可达 IK、串口异常都会暂停新指令并尝试保持当前位置。网页心跳及独立串口进程均有超时停止机制。暂停和退出会保持位置、关闭连接，不会自动释放力矩；如需手动挪动，请先支撑机械臂，再由设备控制方式释放。

每个会话的配置目标、图像误差、速度建议及真实指令计数保存在 `outputs/sessions/`。原生帧只有采集配对时保存。硬件故障造成的保持命令失败会明确显示，软件保持不能替代机械臂自身的物理急停。

## 代码与结果位置

| 内容                | 文件                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| 统一入口 / 网页启动 | `experiment.py`、`start.ps1`                                                                   |
| 任务 2              | `task2.py`、`visual_servo_week6/hand_eye_workflow.py`                                          |
| 任务 3              | `task3.py`、`task3_core.py`、`task3_simulation.py`                                             |
| 相机、串口与 FK/IK  | `frame_source.py`、`hardware_worker.py`、`robot_process.py`、`scs215_bus.py`、`robot_model.py` |
| 网页                | `experiment_web.py`、`experiment_console.html`                                                 |
| 实物参数            | `configs/experiment.yaml`、`configs/robot_model.json`                                          |
| 真实任务 1          | `results/task1/`                                                                               |
| 真实任务 2          | `results/task2/`（等待实验生成）                                                               |
| 采集及仿真          | `data/`、`outputs/`（默认不提交 Git）                                                          |

实现与测试使用了 AI 辅助；提交前应人工核对实际机器人参数与实验结果。算法参考 [OpenCV 相机标定与手眼 API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)、[OpenCV ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) 和 [Chaumette / Hutchinson 视觉伺服教程](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)。
