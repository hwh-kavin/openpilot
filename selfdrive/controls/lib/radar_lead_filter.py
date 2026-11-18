"""BluePilot：雷达点云前车参数处理（视觉车道收敛 + 滤波）。

MPC 前车参数注入（longitudinal_planner）与停车起步判定（Ford 融合块）共用
同一套逻辑：停车时两端使用同一份收敛/滤波后的前车距离与速度；雷达无效或
车道校验拒绝时上层回退 OP 视觉模型前车距离/速度。
"""
from dataclasses import dataclass

from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA

# 车道收敛：保留原有门槛，减少误判带来的抖动。
RADAR_LEAD_LANE_TOLERANCE_M = 1.8
RADAR_LEAD_LANE_EMA = 0.4
# 测量滤波：dRel/vLead EMA 新数据权重
RADAR_LEAD_MEAS_EMA = 0.5


@dataclass
class FilteredLead:
  """收敛+滤波后的前车参数（MPC 与起步判定共用）。"""
  status: bool = False
  dRel: float = 0.0
  vRel: float = 0.0
  vLead: float = 0.0
  aLeadK: float = 0.0
  aLeadTau: float = 1.5
  modelProb: float = 0.0
  radar: bool = True
  radarTrackId: int = -1


def get_vision_lead(sm):
  """OP 视觉模型前车（prob>=0.5 且距离>0.5m），无则 None。"""
  valid = getattr(sm, 'valid', None)
  if sm is not None and (valid is None or valid.get('modelV2', False)):
    vleads = sm['modelV2'].leadsV3
    if len(vleads) > 0:
      vlead = vleads[0]
      if vlead.prob >= 0.5 and vlead.x[0] > 0.5:
        return vlead
  return None


def lead_from_vision(vlead, v_ego: float = 0.0):
  """视觉模型前车 → FilteredLead（无前车时 status=False）。

  与 radard.get_RadarState_from_vision 保持同一语义，否则 MPC 拿到的是“另一种坐标系”：
  - dRel 换算到雷达坐标系（减去 RADAR_TO_CAMERA）；此前漏减，导致恒定高估 1.52 m 距离
  - vRel = vLead - v_ego；此前使用 dataclass 默认值 0，相对速度/TTC 类逻辑全部失效
  - aLeadTau 取 0.3：视觉 lead 的加速度预测衰减比雷达轨迹（1.5）更快
  """
  if vlead is None:
    return FilteredLead()
  v_lead = float(vlead.v[0])
  return FilteredLead(status=True,
                      dRel=max(0.0, float(vlead.x[0]) - RADAR_TO_CAMERA),
                      vRel=v_lead - float(v_ego),
                      vLead=v_lead,
                      aLeadK=float(vlead.a[0]) if len(vlead.a) > 0 else 0.0,
                      aLeadTau=0.3, modelProb=float(vlead.prob), radar=False)


class RadarLeadFilter:
  """雷达点云前车参数处理状态机。

  - 视觉车道收敛：雷达 yRel（左正）与视觉 lead y（右正）横向差 EMA 收敛后
    超过半车道宽判为相邻车道目标、拒绝（回退视觉）。无可靠视觉 lead 可比对
    时保留雷达结果（不拒绝），避免误杀。
  - 测量滤波：dRel/vLead EMA 平滑，换跟踪目标（radarTrackId）立即重置。
  """

  def __init__(self):
    self._lane_diff_ema = 0.0
    self._drel_filt = 0.0
    self._vlead_filt = 0.0
    self._track_id = -1

  def update(self, radar_lead, vision_lead) -> FilteredLead | None:
    """返回收敛+滤波后的雷达 lead；雷达无效/车道拒绝返回 None（上层回退视觉）。"""
    if radar_lead is None:
      return None
    if vision_lead is not None:
      diff = abs(float(radar_lead.yRel) + float(vision_lead.y[0]))
      self._lane_diff_ema = ((1.0 - RADAR_LEAD_LANE_EMA) * self._lane_diff_ema +
                             RADAR_LEAD_LANE_EMA * diff)
      if self._lane_diff_ema >= RADAR_LEAD_LANE_TOLERANCE_M:
        self._track_id = -1
        return None

    if self._track_id != radar_lead.radarTrackId:
      self._track_id = radar_lead.radarTrackId
      self._drel_filt = float(radar_lead.dRel)
      self._vlead_filt = float(radar_lead.vLead)
    else:
      self._drel_filt = ((1.0 - RADAR_LEAD_MEAS_EMA) * self._drel_filt +
                         RADAR_LEAD_MEAS_EMA * float(radar_lead.dRel))
      self._vlead_filt = ((1.0 - RADAR_LEAD_MEAS_EMA) * self._vlead_filt +
                          RADAR_LEAD_MEAS_EMA * float(radar_lead.vLead))
    return FilteredLead(status=True, dRel=self._drel_filt,
                        vRel=float(radar_lead.vRel), vLead=self._vlead_filt,
                        aLeadK=float(radar_lead.aLeadK),
                        aLeadTau=float(radar_lead.aLeadTau),
                        modelProb=float(radar_lead.modelProb),
                        radar=True, radarTrackId=self._track_id)
