# CarLife Companion ↔ openpilot 通信协议

> 版本：**`1.4`**（`schemaVersion: 4`）  
> 方向：手机 APP → openpilot（单向）  
> 用途：高德导航辅助信息旁路显示 + 地图镜像；**不接入**转向/制动控制闭环

对齐源码：

| 文件 | 内容 |
|------|------|
| `carlife-companion/.../model/MapDataPacket.kt` | 通道 A JSON schema |
| `carlife-companion/.../network/DataTransmitter.kt` | UDP JSON 发送 |
| `carlife-companion/.../network/VideoStreamer.kt` | 通道 B CLVF 视频 |
| `HYBRID_MATCH.md` | 固定 ROI / 色标灯 / 车模板 / 绿路径提取 |
| `LABEL_GUIDE.md` | 检测画框（可选补充） |

---

## 0. 双通道总览

| 通道 | 内容 | 默认端口 | 编码 | 频率 |
|------|------|----------|------|------|
| **A 数据** | 导航结构化字段 + 自车/绿路径 | **8888** / UDP | UTF-8 JSON | ~10–20 Hz（推理） |
| **B 视频** | 地图界面镜像 | **8889** / UDP | MJPEG（CLVF 分片） | **~20 Hz**（`streamFps`） |

```
MediaProjection
  ├─ 混合提取 / CV / OCR ──► UDP 广播 :8888  MapDataPacket
  └─ JPEG 缩放 ───────────► UDP 广播 :8889  CLVF 帧
```

### 传输模式（v1.4.2）

| 模式 | 说明 |
|------|------|
| **单播（默认，推荐）** | 发往 APP 中填写的 **OP IP:8888**（JSON）与 **:8889**（视频） |
| 子网广播（可选） | 设置里勾选后发往 `x.x.x.255`；**许多路由器 AP 隔离会导致广播到不了**（ping 仍可能通） |

- OP IP：由 Companion 主界面 / 设置填写，**无固定默认 IP**  
- 视频端口 = 数据端口 + 1  
- 时间基：Unix 毫秒（JSON `timestamp` ↔ 视频 `timestampMs`）  
- OP：`carlifed` `bind 0.0.0.0:8888/8889`；手机与车机须同一 Wi‑Fi 且**未隔离客户端**

---

## 1. 通道 A — MapDataPacket（UDP JSON）

### 1.1 传输

| 项 | 约定 |
|----|------|
| 协议 | UDP **单播**（可选子网广播） |
| 目标 | APP 设定的 OP IP `:8888`（JSON）/ `:8889`（视频） |
| 帧 | **1 UDP 包 = 1 个 JSON 对象** |
| 上限 | **2048** 字节；超长截断 → **非法 JSON 必须丢弃** |
| 可靠 | 无 ACK；以最新有效包为准 |

### 1.2 JSON 约定

| 项 | 约定 |
|----|------|
| 命名 | **camelCase** |
| 默认值 | 始终发出（`encodeDefaults`） |
| null | 可空字段未识别时发 `null`（`explicitNulls`） |
| 兼容 | 忽略未知字段；按 `schemaVersion` 解析 |

### 1.3 完整示例（v4）

```json
{
  "schemaVersion": 4,
  "timestamp": 1722096000000,
  "laneCount": 4,
  "lanes": [
    { "index": 0, "directions": ["left"], "highlighted": false },
    { "index": 1, "directions": ["straight"], "highlighted": true },
    { "index": 2, "directions": ["straight"], "highlighted": true },
    { "index": 3, "directions": ["right"], "highlighted": false }
  ],
  "currentLane": 1,
  "recommendedLanes": [1, 2],
  "lightStatus": "red",
  "countdown": 31,
  "intersectionDistance": 2600.0,
  "turnDirection": "right",
  "laneChange": "none",
  "speedLimit": -1,
  "curveCurvature": null,
  "laneDirection": "straight",
  "speedAction": "decelerate",
  "egoCar": { "x": 0.50, "y": 0.72 },
  "navPath": [
    { "x": 0.50, "y": 0.68 },
    { "x": 0.49, "y": 0.55 },
    { "x": 0.48, "y": 0.40 }
  ],
  "confidence": 0.90,
  "lightConfidence": 0.95,
  "speedConfidence": 0.0,
  "laneConfidence": 0.92,
  "actionConfidence": 0.88,
  "pathConfidence": 0.85,
  "isOccluded": false
}
```

对应画面语义示例：顶栏「2.6km 右转」、4 车道（2/3 道高亮直行）、近车红灯 31s、左侧「减速」、自车 + 车头向前绿路径。

### 1.4 空包 / 心跳

```json
{
  "schemaVersion": 4,
  "timestamp": 1722096000000,
  "laneCount": -1,
  "lanes": [],
  "currentLane": -1,
  "recommendedLanes": [],
  "lightStatus": "unknown",
  "countdown": -1,
  "intersectionDistance": null,
  "turnDirection": "unknown",
  "laneChange": "none",
  "speedLimit": -1,
  "curveCurvature": null,
  "laneDirection": "unknown",
  "speedAction": "unknown",
  "egoCar": null,
  "navPath": [],
  "confidence": 0.0,
  "lightConfidence": 0.0,
  "speedConfidence": 0.0,
  "laneConfidence": 0.0,
  "actionConfidence": 0.0,
  "pathConfidence": 0.0,
  "isOccluded": true
}
```

### 1.5 字段表

#### 元数据

| 字段 | 类型 | 说明 |
|------|------|------|
| `schemaVersion` | int | 当前 **4**；OP 可按版本分支 |
| `timestamp` | long | Unix ms |
| `isOccluded` | bool | 遮挡/不可靠；建议 State Hold |
| `confidence` | float | 综合 0~1 |
| `lightConfidence` / `speedConfidence` / `laneConfidence` / `actionConfidence` / `pathConfidence` | float | 分项 0~1 |

#### 车道

| 字段 | 类型 | 哨兵 | 说明 |
|------|------|------|------|
| `laneCount` | int | `-1` | 总车道数 |
| `lanes` | `LaneInfo[]` | `[]` | 从左到右 |
| `lanes[].index` | int | — | 0-based |
| `lanes[].directions` | string[] | — | 可多选方向 |
| `lanes[].highlighted` | bool | `false` | 顶栏是否高亮推荐 |
| `currentLane` | int | `-1` | 自车所在车道（若可判） |
| `recommendedLanes` | int[] | `[]` | 高亮车道索引；可与 `highlighted` 一致 |
| `laneDirection` | string | `unknown` | HUD 单值主方向 |

#### 灯与路口

| 字段 | 类型 | 哨兵 | 说明 |
|------|------|------|------|
| `lightStatus` | string | `unknown` | **离自车最近**的那组灯 |
| `countdown` | int | `-1` | 秒 |
| `intersectionDistance` | float? | `null` | 米（`2.6km`→`2600`） |
| `turnDirection` | string | `unknown` | 前方转向：`left`/`right`/`straight` |

**多灯消歧（发送端必须遵守）：**

1. 检出全部灯胶囊 + `egoCar`  
2. 取框中心距自车最近的灯（及配对倒计时）写入本包  
3. 远处灯不进协议  

无 `egoCar` 时：回退为偏画面下方 / 面积较大的灯（次优）。

#### 导航提示

| 字段 | 类型 | 哨兵 | 说明 |
|------|------|------|------|
| `laneChange` | string | `none` | 变道：`left`/`right`/`none`/`unknown` |
| `speedAction` | string | `unknown` | `decelerate`/`start`/`maintain`/`unknown` |
| `speedLimit` | int | `-1` | 限速 km/h；**不是**车速表读数 |
| `curveCurvature` | float? | `null` | κ (1/m)；正左负右 |

#### 自车与绿路径（屏幕归一化）

| 字段 | 类型 | 哨兵 | 说明 |
|------|------|------|------|
| `egoCar` | `{x,y}`? | `null` | 自车中心，0~1 |
| `navPath` | `{x,y}[]` | `[]` | 绿规划线折线，**近→远**（车头方向），最多 **24** 点 |

坐标系：相对采集帧；原点**左上**；`x` 右、`y` 下；均钳制到 `[0,1]`。

路径语义：

- 仅含与**车头连通**的导航绿带（排除草地等）  
- OP 可按视频分辨率缩放：`px = x * videoWidth`  
- 与通道 B 叠加时，优先用同时间戳最近的 JSON  

### 1.6 枚举汇总

| 用途 | 取值 |
|------|------|
| 方向 | `left` \| `straight` \| `right` \| `uturn` \| `unknown` |
| 灯色 | `red` \| `green` \| `yellow` \| `unknown` |
| 加减速 | `decelerate` \| `start` \| `maintain` \| `unknown` |
| 变道 | `left` \| `right` \| `none` \| `unknown` |
| 转向 | `left` \| `right` \| `straight` \| `unknown` |

---

## 2. 通道 B — 地图视频（UDP CLVF / MJPEG）

与 v1.2 相同，要点：

| 项 | 约定 |
|----|------|
| 端口 | 8889（数据+1） |
| 编码 | JPEG，默认最大宽 540、质量 ~50 |
| 帧率 | 默认 20 Hz |
| 头 | 30 字节大端 `CLVF`，见下表 |
| 单包 | ≤ 1400 字节 |

| 偏移 | 长度 | 字段 |
|------|------|------|
| 0 | 4 | `CLVF` |
| 4 | 1 | `version=1` |
| 5 | 1 | `codec=1` JPEG |
| 6 | 2 | `flags=0` |
| 8 | 4 | `frameSeq` |
| 12 | 8 | `timestampMs` |
| 20 | 2 | `width` |
| 22 | 2 | `height` |
| 24 | 2 | `fragIndex` |
| 26 | 2 | `fragCount` |
| 28 | 2 | `payloadLen` |
| 30 | N | JPEG 分片 |

收齐分片 → JPEG → 显示；与 JSON 按时间戳对齐。详细伪代码见历史版本或 `VideoStreamer.kt`。

---

## 3. OP 侧使用建议

1. **HUD**：灯色/倒计时、限速、减速、车道高亮、转向距离  
2. **叠线**：`navPath` 画在视频上（需有完整视频帧尺寸）  
3. **State Hold**：`isOccluded` 或关键字段 `unknown` 时保持上一有效值短时  
4. **绝不**把本协议写入横向/纵向控制  
5. `schemaVersion < 4`：无 `navPath`/`egoCar`/`turnDirection` 等则当缺省哨兵  

### 接收骨架

```python
import json, socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 8888))

while True:
    raw, _ = sock.recvfrom(4096)
    try:
        pkt = json.loads(raw.decode("utf-8"))
    except Exception:
        continue
    ver = pkt.get("schemaVersion", 1)
    light = pkt.get("lightStatus", "unknown")
    countdown = pkt.get("countdown", -1)
    dist = pkt.get("intersectionDistance")
    turn = pkt.get("turnDirection", "unknown")
    action = pkt.get("speedAction", "unknown")
    ego = pkt.get("egoCar")          # {"x","y"} | None
    path = pkt.get("navPath") or []  # [{"x","y"}, ...]
    lanes = pkt.get("lanes") or []
    # → cereal / UI
```

---

## 4. 提取方式与协议字段映射（实现参考）

| 字段 | 推荐提取 |
|------|----------|
| 顶栏距离/转向 | 固定 ROI + OCR → `intersectionDistance` + `turnDirection` |
| 顶栏车道条 | 固定 ROI → `lanes` + `highlighted` / `recommendedLanes` |
| 加减速 | 固定 ROI / 模板 → `speedAction` |
| 红绿灯+秒数 | 色标搜索或多目标检测；近车优选 → `lightStatus`/`countdown` |
| 自车 | `dataset/car` 多模板匹配 → `egoCar` |
| 绿路径 | 车头种子连通 HSV 分割 → `navPath` |
| 视频 | MediaProjection → 通道 B |

详见 `HYBRID_MATCH.md`。

---

## 5. APP 填充进度（v1.4）

| 字段 | 状态 |
|------|------|
| 通道 A/B 发送骨架 | ✅ |
| `schemaVersion` / 灯 / 倒计时 / 限速 / `laneDirection` | ✅ 基线 |
| `speedAction` 及分项 confidence 字段 | ✅ 字段已发；识别待接 |
| `lanes` 完整多车道 + `highlighted` | ⚠️ 仍多为单箭头占位 |
| `turnDirection` / `laneChange` / `intersectionDistance` | ⚠️ 字段已定义，多发哨兵 |
| `egoCar` / `navPath` | ⚠️ 协议已定义；PC 侧 `hybrid_extract` 已验证，APP 待接入 |
| `curveCurvature` | ❌ 常 `null` |

---

## 6. 修订记录

| 版本 | schemaVersion | 说明 |
|------|---------------|------|
| 1.0 | — | 限速 / 灯 / 倒计时 / 单方向 |
| 1.1 | — | 多车道、路口距离、曲率 |
| 1.2 | — | 通道 B CLVF 视频 |
| 1.3 | — | `speedAction` + 分项置信度 |
| **1.4** | **4** | `schemaVersion`；`LaneInfo.highlighted`；`recommendedLanes`；`turnDirection`；`laneChange`；`egoCar`；`navPath`（车头绿路径）；多灯近车消歧；UDP 上限 2048 |
| **1.4.1** | **4** | 曾默认子网广播；可选单播 |
| **1.4.2** | **4** | **默认单播**到 APP 填写的 OP IP（`:8888`/`:8889`）；无固定默认 IP；广播仅作选项（AP 隔离下无效） |
