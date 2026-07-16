export interface PanelCondition {
  allConditionsTrue?: PanelCondition[]
  anyConditionsTrue?: PanelCondition[]
  notCondition?: PanelCondition
  paramIsTrue?: string
  paramIsFalse?: string
  paramValueEquals?: Record<string, string | number | boolean>
  paramValueGreaterThan?: Record<string, number>
  paramValueLessThan?: Record<string, number>
  paramExists?: string
  isOnroad?: boolean
  isOffroad?: boolean
  isEngaged?: boolean
  hasCarParams?: boolean
  hasLongitudinalControl?: boolean
  hasBlindSpotMonitoring?: boolean
  hasIntelligentCruiseButtonManagement?: boolean
  hasAlphaLongitudinalAvailable?: boolean
  isAngleSteering?: boolean
  isMadsLimitedBrand?: boolean
  isPcmCruise?: boolean
  isICBMAvailable?: boolean
  isReleaseBranch?: boolean
  isTestedBranch?: boolean
  isDevelopmentBranch?: boolean
  brandEquals?: string
  teslaHasVehicleBus?: boolean
  hyundaiAlphaLongAvailable?: boolean
  hasStopAndGo?: boolean
  subaruHasSng?: boolean
  torqueAllowed?: boolean
  isSpRelease?: boolean
  capabilityEquals?: { field: string; equals: string | number | boolean }
  reason?: string
}

export interface PanelState {
  isOnroad: boolean
  isOffroad: boolean
  hasCarParams: boolean
  hasLongitudinalControl?: boolean
  hasBlindSpotMonitoring?: boolean
  hasIntelligentCruiseButtonManagement?: boolean
  hasAlphaLongitudinalAvailable?: boolean
  isAngleSteering?: boolean
  isMadsLimitedBrand?: boolean
  isPcmCruise?: boolean
  isICBMAvailable?: boolean
  isReleaseBranch?: boolean
  isTestedBranch?: boolean
  isDevelopmentBranch?: boolean
  isEngaged?: boolean
  teslaHasVehicleBus?: boolean
  hyundaiAlphaLongAvailable?: boolean
  hasStopAndGo?: boolean
  subaruHasSng?: boolean
  torqueAllowed?: boolean
  isSpRelease?: boolean
  brandEquals?: Record<string, boolean>
  capabilities?: Record<string, string | number | boolean>
}

export interface SelectionOption {
  name: string
  value: string | number
  desc?: string
  default?: boolean
  enableConditions?: PanelCondition
}

interface BaseControl {
  type: string
  title: string
  desc?: string
  hidden?: boolean
  webSupported?: boolean
  enableConditions?: PanelCondition
  visibleConditions?: PanelCondition
  dynamic_desc?: boolean
  descriptions?: Record<string, string>
  description_conditions?: Record<string, PanelCondition>
  dynamic_title?: boolean
  titles?: { enabled?: string; disabled?: string }
  dynamic_styling?: boolean
  styles?: { enabled?: Record<string, string>; disabled?: Record<string, string> }
  button_style?: Record<string, string> & {
    background_color?: string
    text_color?: string
  }
}

export interface ToggleControl extends BaseControl {
  type: 'toggle'
  param: string
  needsOnroadCycle?: boolean
  confirm?: boolean
  confirmation?: boolean
  confirm_text?: string
  confirm_yes_text?: string
  confirm_no_text?: string
}

export interface SelectionControl extends BaseControl {
  type: 'selection'
  param: string
  options: SelectionOption[]
  unit?: string
  unitMetric?: string
}

export interface SegmentedControl extends BaseControl {
  type: 'segmented_control'
  param: string
  options: SelectionOption[]
  showDescBottom?: boolean
}

export interface IntegerControl extends BaseControl {
  type: 'integer'
  param: string
  min: number
  max: number
  increment: number
  division?: number
  unit?: string
  unitMetric?: string
}

export interface FloatControl extends BaseControl {
  type: 'float'
  param: string
  min: number
  max: number
  increment: number
  unit?: string
  unitMetric?: string
}

export interface CommandButtonControl extends BaseControl {
  type: 'command_button'
  param?: string
  params?: string[]
  value?: string | number | boolean
  button_text?: string
  action?: string
  confirm?: boolean
  confirm_text?: string
  confirm_yes_text?: string
  confirm_no_text?: string
  confirm_button_text?: string
  cancel_button_text?: string
  device_only_message?: string
}

export interface FileViewerControl extends BaseControl {
  type: 'file_viewer'
  path: string
  button_text?: string
  header?: string
}

export interface StaticTextControl extends BaseControl {
  type: 'static_text'
}

export interface StaticParamDisplayControl extends BaseControl {
  type: 'static_param_display'
  param: string
}

export interface PlatformDisplayControl extends BaseControl {
  type: 'platform_display'
  value_param: string
  value_color?: string
}

export interface RecentChangesControl extends BaseControl {
  type: 'recent_changes'
}

export interface RestartUIControl extends BaseControl {
  type: 'restart_ui'
  button_text?: string
  confirm?: boolean
  confirm_text?: string
  confirm_yes_text?: string
  confirm_no_text?: string
}

export interface UnsupportedControl extends BaseControl {
  type: 'file_param_display' | 'param_viewer' | 'param_list_viewer'
  param?: string
}

export type PanelControl =
  | ToggleControl
  | SelectionControl
  | SegmentedControl
  | IntegerControl
  | FloatControl
  | CommandButtonControl
  | FileViewerControl
  | StaticTextControl
  | StaticParamDisplayControl
  | PlatformDisplayControl
  | RecentChangesControl
  | RestartUIControl
  | UnsupportedControl

export interface PanelGroup {
  groupName: string
  title: string
  groupDescription?: string
  controls: PanelControl[]
  allowReset?: boolean
  hidden?: boolean
  enableResetButton?: boolean
}

export interface PanelConfig {
  menuName: string
  menuDescription?: string
  menuIcon?: string
  groups: PanelGroup[]
}

export interface PanelMetadata {
  id: string
  name: string
  description?: string
  icon?: string
}

export interface PanelsListResponse {
  success: boolean
  panels: PanelMetadata[]
}

export interface PanelResponse {
  success: boolean
  panel?: PanelConfig
}

export interface PanelStateResponse {
  success: boolean
  state?: PanelState
}

export interface PanelCommandRequest {
  action: string
  param?: string
  value?: string | number | boolean
  params?: string[]
  username?: string
  password?: string
  api_key?: string
  security_js_code?: string
  web_service_key?: string
  remove?: boolean
}

export interface PanelCommandResponse {
  success: boolean
  error?: string
  message?: string
  has_keys?: boolean
  username?: string
  content?: string
  modified?: string
  [key: string]: unknown
}
