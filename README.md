<a href="https://opendrivelab.com/EgoHumanoid-V2/#overview">
  <img src="assets/teaser.gif" width="1280" alt="EgoHumanoid-V2 animated paper teaser: robot-free human demonstrations, visual and coarse-to-fine action alignment, policy learning, and zero-shot whole-body loco-manipulation">
</a>

<h1 align="center">EgoHumanoid-V2</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2609.37181-b31b1b)](https://arxiv.org/abs/2609.37181)
[![Project Page](https://img.shields.io/badge/Project-Page-6b46c1)](https://opendrivelab.com/EgoHumanoid-V2/)
[![Video](https://img.shields.io/badge/Video-Overview-ea4335)](https://opendrivelab.com/EgoHumanoid-V2/assets/videos/project-overview-v14.mp4)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Homepage](https://img.shields.io/badge/Homepage-Jin%20Chen-2563eb)](https://jinchen0056.github.io/)
[![Email](https://img.shields.io/badge/Contact-Email-6b46c1)](mailto:cj65000816081@gmail.com)

</div>

<p align="center">
  <strong>🤖 Human-to-humanoid skill transfer for coordinated whole-body loco&#8209;manipulation.</strong>
</p>

Human demonstrations capture rich whole-body motion across diverse scenes without robot teleoperation. **EgoHumanoid-V2** transfers these coordinated skills to a humanoid through coarse-to-fine action alignment: kinematic reference correction improves motion targets, and dynamics-aware refinement improves their execution through a frozen whole-body controller.

Visual alignment replaces human arms with rendered robot arms, while training-time image augmentation improves viewpoint robustness. A **π₀.₅ vision-language-action policy** learns from the aligned demonstrations and executes through **Unitree G1 + BrainCo Revo2 + the SONIC whole-body controller**. The paper evaluates zero-shot transfer on four real-world tasks without target-task robot demonstrations.

---

<p></p>

<a id="todo"></a>

### Workflow & Release Checklist

> **Coming soon:** The remaining unchecked items are expected to be released within the next two weeks. Stay tuned!

1. **Hardware Setup**
   - [x] [Robot Hardware](#robot-hardware)
   - [x] [Human Hardware](#human-data-collection-hardware)
2. **Environment Setup**
   - [x] [Robot Control](#robot-runtime-environment)
   - [x] [Human Data Collection](#human-collection-environment)
   - [x] [Vision Alignment](#vision-alignment-environment)
   - [x] [Policy Training & Serving](#policy-environment)
3. **Data Collection**
   - [x] [Human Collection](#human-data-collection)
4. **Data Alignment**
   - [x] **[Action Alignment](#action-alignment)** ⭐
   - [x] [Vision Alignment](#vision-alignment)
5. **Model Training**
   - [ ] Training pipeline
6. **Policy Deployment**
   - [ ] Deployment pipeline

---

<p></p>

<a id="hardware-setup"></a>

## 🛠️ Hardware Setup

<table width="100%">
<tr>
<td width="50%" valign="top">
<a id="human-data-collection-hardware"></a>
<h3>Human Hardware</h3>
<img src="assets/hardware-human.gif" width="100%" alt="Animated human hardware tour: PICO headset, five body trackers and neck-mounted GoPro">
<details>
<summary>🔴 <strong>PICO 4 Ultra</strong></summary>

- **Tracking:** [PICO 4 Ultra](https://www.picoxr.com/global/products/pico4-ultra) + trackers<br>for whole-body and hand tracking
- **Software:** XRobo&#8203;Toolkit<br>[Setup&nbsp;guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md#step-1-install-xrobotoolkit) · [Python&nbsp;SDK](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind)
- **Connec&#8203;tion:** same LAN as the PC<br>enable Head, Hand and Send

</details>
<br>
<details>
<summary>🔵 <strong>5× PICO Trackers</strong></summary>

- **Placement:** five trackers<br>both wrists, waist and both ankles
- **Calibration:** PICO Motion Tracker<br>pair and calibrate in the app ([guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md#motion-tracker-calibration))
- **Mode:** Full Body in XRobo&#8203;Toolkit<br>enable Hand tracking ([setup&nbsp;guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md#step-4-connect-the-pico-to-your-workstation))

</details>
<br>
<details>
<summary>🟠 <strong>GoPro</strong></summary>

- **Mount:** neck-mounted bracket<br>Example: [JD listing](https://item.jd.com/product/5qVT4d95gaChewMJeYVDKQ.html)
- **Alignment:** match height, pitch and framing to the robot camera
- **Connec&#8203;tion:** Bluetooth pairing and WiFi setup;<br>WiFi / COHN recording and download ([API](https://github.com/gopro/OpenGoPro))

</details>
</td>
<td width="50%" valign="top">
<a id="robot-hardware"></a>
<h3>Robot Hardware</h3>
<img src="assets/hardware-robot.gif" width="100%" alt="Animated robot hardware tour: Unitree G1, BrainCo Revo2 hands and head-mounted GoPro">
<details>
<summary>🟢 <strong>Unitree G1</strong></summary>

- **Platform:** [Unitree G1](https://www.unitree.com/g1/), 29 body DOF<br>excluding hands ([joint&nbsp;layout](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/resources/robots/g1_description/README.md))
- **Control:** [SONIC whole-body controller](https://github.com/NVlabs/GR00T-WholeBodyControl)<br>runs on an external Linux PC
- **Connec&#8203;tion:** wired or WiFi to the PC<br>via the ZMQ bridge

</details>
<br>
<details>
<summary>🔵 <strong>BrainCo Revo2</strong></summary>

- **Hand:** 6 active DOF per hand<br>11 including passive joints ([specs](https://brainco.tech/docs/revolimb-hand/en/revo2/parameters.html))
- **Software:** [official&nbsp;SDK](https://github.com/BrainCoTech/brainco-hand-sdk)<br>hand control and state feedback
- **Mount:** custom G1–Revo2 adapters<br>STL: [part 1](hardware/Revo2-G1-1.stl) · [part 2](hardware/Revo2-G1-2.stl)

</details>
<br>
<details>
<summary>🟠 <strong>GoPro</strong></summary>

- **Mount:** head-mounted bracket<br>STL: [part 1](hardware/G1_GoPro_1.stl) · [part 2](hardware/G1_GoPro_2.stl)
- **Alignment:** match height, pitch and framing to the human camera
- **Connec&#8203;tion:** USB control and video to robot;<br>wired / WiFi forwarding to host ([API](https://github.com/gopro/OpenGoPro))

</details>

</td>
</tr>
</table>

<p></p>

<a id="environment-setup"></a>

## 💻 Environment Setup

Install the environments needed for your workflow below. Action alignment uses the Robot Control environment; vision alignment has its own installer.

<a id="prerequisites"></a>

<a id="installation"></a>

<a id="robot-runtime-environment"></a>

<details>
<summary>🤖 <strong>Robot Control</strong></summary>

### Host

Ubuntu 22.04 · NVIDIA driver **580+**.

**1. Install dependencies.**

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl build-essential cmake git git-lfs libgl1 libegl1 libegl-mesa0 libglib2.0-0 ffmpeg libyaml-cpp-dev libeigen3-dev libmsgpack-dev libzmq3-dev nlohmann-json3-dev libssl-dev zlib1g-dev
curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh
source "$HOME/.local/bin/env"
curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends cuda-minimal-build-12-4
```

**2. Install TensorRT and ONNX Runtime.** Download `TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0.tar.gz` from [NVIDIA](https://developer.nvidia.com/tensorrt/download) to the repository root, then run:

```bash
sudo tar -xzf TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0.tar.gz -C /opt
curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz -o /tmp/onnxruntime.tgz
sudo tar -xzf /tmp/onnxruntime.tgz -C /opt
```

**3. Install environment.** Repeat the exports when opening a new terminal.

```bash
export TensorRT_ROOT=/opt/TensorRT-10.13.3.9
export onnxruntime_ROOT=/opt/onnxruntime-linux-x64-1.16.3
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$onnxruntime_ROOT/lib:$CUDAToolkit_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash setup/robot/install_host.sh
```

**4. Activate environment.**

```bash
source sonic/.venv/bin/activate
```

### Robot

ARM64 · Ubuntu 20.04 · ROS 2 Foxy · Python 3.8.

**1. Install environment.**

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv build-essential cmake git python3-colcon-common-extensions ros-foxy-rmw-cyclonedds-cpp ros-foxy-rosidl-generator-dds-idl libyaml-cpp-dev libssl-dev libgl1 libglib2.0-0
bash setup/robot/install_robot.sh
```

**2. Activate environment.**

```bash
source /opt/ros/foxy/setup.bash
source .venv-robot/ros_ws/install/setup.bash
source .venv-robot/bin/activate
```

</details>
<br>
<a id="human-collection-environment"></a>

<details>
<summary>👤 <strong>Human Data Collection</strong></summary>

<p></p>
<p>Ubuntu 22.04 · Python 3.10.</p>

**1. Install environment.**

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates python3.10-dev python3.10-venv build-essential cmake git ffmpeg bluez network-manager
bash setup/human/install.sh
```

**2. Activate environment.**

```bash
source .venv-human/bin/activate
```

</details>
<br>
<a id="vision-alignment-environment"></a>

<details>
<summary>🎨 <strong>Vision Alignment</strong></summary>

<p></p>

Tested on Linux with a 48 GB NVIDIA GPU. Allow about 35 GB of disk space. Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), FFmpeg, and an NVIDIA driver compatible with CUDA 13.0; reuse these if already installed.

**1. Obtain model access.** Request access to [SAM 3.1](https://huggingface.co/facebook/sam3.1), then authenticate:

```bash
uvx --from huggingface_hub hf auth login
```

**2. Install the environments and models.**

```bash
bash setup/vision/install.sh
```

The installer creates two isolated environments under `runtime/vision`: Python 3.12 for segmentation/rendering and Python 3.10 for inpainting. It downloads pinned models and verifies their checksums. The alignment command calls both environments automatically; no manual switching is needed.

Installation scripts, dependency pins, model manifests, and the SAM compatibility patch are in [`setup/vision/`](setup/vision/). Processing code, tests, and example motion files are in [`data_alignment/vision_alignment/`](data_alignment/vision_alignment/).

</details>
<br>

<a id="policy-environment"></a>

<details>
<summary>🧠 <strong>Policy Training &amp; Serving</strong></summary>

<p></p>
<p>Ubuntu 22.04 · NVIDIA GPU and driver.</p>

**1. Install dependencies.**

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl git build-essential libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh
source "$HOME/.local/bin/env"
```

**2. Install [OpenPI](https://github.com/Physical-Intelligence/openpi#installation).**

```bash
bash setup/policy/install.sh
```

**3. Activate environment.**

```bash
source openpi/.venv/bin/activate
```

</details>

<p></p>

<a id="data-collection"></a>

## 🎥 Data Collection

<a id="human-data-collection"></a>

### 👤 Human Data Collection

**1. Prepare devices.**

Connect the host and PICO headset to the same network. Start the XRoboToolkit PC service on the host and the XRoboToolkit app on the headset, then enable body and hand tracking. Turn on the GoPro, enable Bluetooth pairing and select Video mode; the collector connects it to the host's WiFi network in the next step.

**2. Start recording.**

```bash
eh2-collect-human --task-name "{task_name}" --task "{task_instruction}"
```

Open `http://127.0.0.1:8080`. Enter the host's WiFi credentials if needed, then click **Prepare collection → Start recording → Save episode / Discard episode**. Wait for saving before recording again. Episodes are saved in LeRobot format to `data/{task_name}/`; new recordings append to the dataset.

**3. Check and synchronize data.**

```bash
eh2-sync-human --root "data/{task_name}"
```

Checks the recordings, synchronizes motion and video at 50 Hz, then checks the output. Results are saved to `data/{task_name}_synced/`, with synchronization and data quality reports in `meta/`.

<p></p>

<a id="data-processing"></a>
<a id="data-alignment"></a>

## ⚡ Data Alignment

<a id="action-alignment"></a>

### 🦾 Action Alignment

We provide one example trajectory for each of the four tasks in [examples](data_alignment/action_alignment/examples). The simulation replays below compare these trajectories before (left) and after (right) alignment.

<table>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/ea461aca-9635-46b3-b522-6a3471bf7dc6

</td>
<td width="50%">

https://github.com/user-attachments/assets/68c5a826-4416-4180-a525-5b584e38e250

</td>
</tr>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/906522e9-1088-4619-9d0f-6d2844623d29

</td>
<td width="50%">

https://github.com/user-attachments/assets/71c1de99-8f2e-4ee1-875c-fe4be36db618

</td>
</tr>
</table>

**1. Align motion.**

Place the input recording at `{task_name}/input.parquet`. Set `{endpoints}` to `hands` for hand alignment or `feet` for foot alignment. Pretrained hand and foot response MLP weights are included in [models/response](data_alignment/action_alignment/models/response).

```bash
eh2-align \
  --input "{task_name}/input.parquet" \
  --endpoints {endpoints} \
  --output "results/{task_name}"
```

Results are saved under `results/{task_name}/`: `data.parquet` contains the human SMPL motion and aligned tokens; `run.json` records per-round metrics and the selected round.

**2. Replay motion.**

**SMPL replay:** encode the human motion online to simulate before alignment.

```bash
eh2-replay \
  --input "{task_name}/data.parquet" \
  --mode smpl \
  --output "results/{task_name}/before"
```

**Token replay:** replay the aligned motion tokens to simulate after alignment.

```bash
eh2-replay \
  --input "{task_name}/data.parquet" \
  --mode token \
  --output "results/{task_name}/after"
```

Each command saves `replay.mp4` in its output directory. Results may vary slightly between runs.

<a id="vision-alignment"></a>

### 🎨 Vision Alignment

The pipeline replaces human arms with rendered **G1 arms** while preserving the original gloves and hand–object contact. We upgrade the paper's **SAM 2** to **[SAM 3.1](https://github.com/facebookresearch/sam3)** for more robust arm and wrist-tracker segmentation. **[DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser)** reconstructs the background.

Synchronized joint trajectories drive the G1 model through an approximate virtual camera. Wrist locations and segmentation masks then refine the rendered arms' placement in the image without changing the joint angles.

The examples below demonstrate visual alignment for closing a laptop, opening a fridge, and pulling a curtain. Each row shows **RGB → Mask → Inpainting → Robot-arm rendering**. Download the video and motion files to reproduce the results.

**Close Laptop** · [Video](https://github.com/user-attachments/assets/aee7625e-2b8a-4de2-9e74-f549cf8324f6) · [Motion](https://github.com/OpenDriveLab/EgoHumanoid-V2/raw/refs/heads/main/data_alignment/vision_alignment/examples/close_laptop/motion.npz)

<table>
<tr>
<th width="25%">Human RGB</th>
<th width="25%">SAM 3.1 Mask</th>
<th width="25%">Inpainting</th>
<th width="25%">Robot-arm Rendering</th>
</tr>
<tr>
<td width="25%">

<img src="assets/vision_alignment/close_laptop/rgb.webp" width="100%" alt="Close Laptop animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/close_laptop/mask.webp" width="100%" alt="Close Laptop animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/close_laptop/inpaint.webp" width="100%" alt="Close Laptop animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/close_laptop/render.webp" width="100%" alt="Close Laptop animation">

</td>
</tr>
</table>

**Open Fridge** · [Video](https://github.com/user-attachments/assets/f6b4d3b5-4d71-417d-8c4b-88fa4739a5a6) · [Motion](https://github.com/OpenDriveLab/EgoHumanoid-V2/raw/refs/heads/main/data_alignment/vision_alignment/examples/open_fridge/motion.npz)

<table>
<tr>
<th width="25%">Human RGB</th>
<th width="25%">SAM 3.1 Mask</th>
<th width="25%">Inpainting</th>
<th width="25%">Robot-arm Rendering</th>
</tr>
<tr>
<td width="25%">

<img src="assets/vision_alignment/open_fridge/rgb.webp" width="100%" alt="Open Fridge animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/open_fridge/mask.webp" width="100%" alt="Open Fridge animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/open_fridge/inpaint.webp" width="100%" alt="Open Fridge animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/open_fridge/render.webp" width="100%" alt="Open Fridge animation">

</td>
</tr>
</table>

**Pull Curtain** · [Video](https://github.com/user-attachments/assets/073e3d1b-ab19-477b-8bf5-5bdd19e8bfb3) · [Motion](https://github.com/OpenDriveLab/EgoHumanoid-V2/raw/refs/heads/main/data_alignment/vision_alignment/examples/pull_curtain/motion.npz)

<table>
<tr>
<th width="25%">Human RGB</th>
<th width="25%">SAM 3.1 Mask</th>
<th width="25%">Inpainting</th>
<th width="25%">Robot-arm Rendering</th>
</tr>
<tr>
<td width="25%">

<img src="assets/vision_alignment/pull_curtain/rgb.webp" width="100%" alt="Pull Curtain animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/pull_curtain/mask.webp" width="100%" alt="Pull Curtain animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/pull_curtain/inpaint.webp" width="100%" alt="Pull Curtain animation">

</td>
<td width="25%">

<img src="assets/vision_alignment/pull_curtain/render.webp" width="100%" alt="Pull Curtain animation">

</td>
</tr>
</table>

**Align a recording.**

Save the video as `{task_name}/rgb.mp4` and its synchronized motion as `{task_name}/motion.npz`. The NPZ contains `body_q` (29 G1 joint angles in radians per frame), `joint_names`, and `fps` matching the video.

```bash
runtime/vision/.venv-sam/bin/eh2-visual-align \
  --video "{task_name}/rgb.mp4" \
  --motion "{task_name}/motion.npz" \
  --runtime "runtime/vision" \
  --output "results/{task_name}"
```

Results are saved under `results/{task_name}/`: `aligned.mp4` contains the aligned video; `run.json` records settings and completed stages.

<p></p>

<a id="model-training"></a>

## 🤖 Model Training

Coming soon.

<p></p>

<a id="deployment"></a>

## 🚀 Policy Deployment

Coming soon.

<p></p>

<a id="citation"></a>

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{chen2026egohumanoidv2,
  title={{EgoHumanoid-V2}: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation},
  author={Chen, Jin and Jiang, Yiming and Xu, Chongyang and Shi, Modi and Peng, Shijia and Chen, Li and Li, Tianyu and Xu, Mu and Chen, Yilun and Hoi, Steven and Li, Hongyang},
  journal={arXiv preprint arXiv:2609.37181},
  year={2026}
}
```

---

<p></p>

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

We sincerely thank the following projects and teams:

| [![OpenPI](https://img.shields.io/badge/OpenPI-Physical%20Intelligence-d58b45)](https://github.com/Physical-Intelligence/openpi) | [![SONIC](https://img.shields.io/badge/SONIC-NVIDIA-76b900)](https://github.com/NVlabs/GR00T-WholeBodyControl) | [![XRoboToolkit](https://img.shields.io/badge/XRoboToolkit-PICO%20VR-397bbf)](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client) | [![Open GoPro](https://img.shields.io/badge/Open%20GoPro-GoPro-009fdf)](https://github.com/gopro/OpenGoPro) |
| :---: | :---: | :---: | :---: |
| VLA&nbsp;models | Whole-body&nbsp;control | PICO&nbsp;tracking | GoPro&nbsp;interface |

---

## Contact

Questions, ideas, or interested in collaborating? We'd love to hear from you—feel free to connect with Jin Chen on WeChat!

<p align="center">
  <a href="assets/wechat.png"><img src="assets/wechat.png" width="240" alt="Scan to add Jinstonn on WeChat"></a>
</p>
