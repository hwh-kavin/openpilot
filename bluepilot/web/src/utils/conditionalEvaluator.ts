import type { PanelCondition, PanelControl, PanelState } from '@/types/panels'

type ParamRecord = Record<string, { value: unknown }>

function getParamValue(params: ParamRecord, key: string): unknown {
  return params[key]?.value
}

function isParamTruthy(value: unknown): boolean {
  return value === true || value === '1' || value === 1
}

function isParamFalsy(value: unknown): boolean {
  return value === false || value === '0' || value === 0 || value == null
}

export function evaluateCondition(
  condition: PanelCondition,
  state: PanelState,
  params: ParamRecord,
): boolean {
  if (condition.allConditionsTrue) {
    return condition.allConditionsTrue.every((child) => evaluateCondition(child, state, params))
  }

  if (condition.anyConditionsTrue) {
    return condition.anyConditionsTrue.some((child) => evaluateCondition(child, state, params))
  }

  if (condition.paramIsTrue !== undefined) {
    return isParamTruthy(getParamValue(params, condition.paramIsTrue))
  }

  if (condition.paramIsFalse !== undefined) {
    return isParamFalsy(getParamValue(params, condition.paramIsFalse))
  }

  if (condition.paramValueEquals) {
    return Object.entries(condition.paramValueEquals).every(([key, expected]) => {
      return String(getParamValue(params, key)) === String(expected)
    })
  }

  if (condition.paramValueGreaterThan) {
    return Object.entries(condition.paramValueGreaterThan).every(([key, threshold]) => {
      const raw = getParamValue(params, key)
      const value = typeof raw === 'number' ? raw : parseFloat(String(raw))
      const limit = typeof threshold === 'number' ? threshold : parseFloat(String(threshold))
      return !Number.isNaN(value) && value > limit
    })
  }

  if (condition.paramValueLessThan) {
    return Object.entries(condition.paramValueLessThan).every(([key, threshold]) => {
      const raw = getParamValue(params, key)
      const value = typeof raw === 'number' ? raw : parseFloat(String(raw))
      const limit = typeof threshold === 'number' ? threshold : parseFloat(String(threshold))
      return !Number.isNaN(value) && value < limit
    })
  }

  if (condition.paramExists !== undefined) {
    return getParamValue(params, condition.paramExists) !== undefined
  }

  if (condition.isOnroad !== undefined) return state.isOnroad === condition.isOnroad
  if (condition.isOffroad !== undefined) return state.isOffroad === condition.isOffroad
  if (condition.hasCarParams !== undefined) return state.hasCarParams === condition.hasCarParams
  if (condition.hasLongitudinalControl !== undefined) {
    return state.hasLongitudinalControl === condition.hasLongitudinalControl
  }
  if (condition.hasBlindSpotMonitoring !== undefined) {
    return state.hasBlindSpotMonitoring === condition.hasBlindSpotMonitoring
  }
  if (condition.hasIntelligentCruiseButtonManagement !== undefined) {
    return state.hasIntelligentCruiseButtonManagement === condition.hasIntelligentCruiseButtonManagement
  }
  if (condition.hasAlphaLongitudinalAvailable !== undefined) {
    return state.hasAlphaLongitudinalAvailable === condition.hasAlphaLongitudinalAvailable
  }
  if (condition.isAngleSteering !== undefined) return state.isAngleSteering === condition.isAngleSteering
  if (condition.isMadsLimitedBrand !== undefined) return state.isMadsLimitedBrand === condition.isMadsLimitedBrand
  if (condition.isPcmCruise !== undefined) return state.isPcmCruise === condition.isPcmCruise
  if (condition.isICBMAvailable !== undefined) return state.isICBMAvailable === condition.isICBMAvailable
  if (condition.isReleaseBranch !== undefined) return state.isReleaseBranch === condition.isReleaseBranch
  if (condition.isTestedBranch !== undefined) return state.isTestedBranch === condition.isTestedBranch
  if (condition.isDevelopmentBranch !== undefined) {
    return state.isDevelopmentBranch === condition.isDevelopmentBranch
  }

  if (condition.brandEquals !== undefined) {
    return state.brandEquals?.[condition.brandEquals] === true
  }

  if (condition.isEngaged !== undefined) {
    return (state.isEngaged ?? false) === condition.isEngaged
  }

  if (condition.teslaHasVehicleBus !== undefined) {
    return state.teslaHasVehicleBus === condition.teslaHasVehicleBus
  }
  if (condition.hyundaiAlphaLongAvailable !== undefined) {
    return state.hyundaiAlphaLongAvailable === condition.hyundaiAlphaLongAvailable
  }
  if (condition.hasStopAndGo !== undefined) return state.hasStopAndGo === condition.hasStopAndGo
  if (condition.subaruHasSng !== undefined) return state.subaruHasSng === condition.subaruHasSng
  if (condition.torqueAllowed !== undefined) return state.torqueAllowed === condition.torqueAllowed
  if (condition.isSpRelease !== undefined) return state.isSpRelease === condition.isSpRelease

  if (condition.capabilityEquals) {
    const { field, equals } = condition.capabilityEquals
    return String(state.capabilities?.[field]) === String(equals)
  }

  if (condition.notCondition) {
    return !evaluateCondition(condition.notCondition, state, params)
  }

  return true
}

export function evaluateConditions(
  conditions: PanelCondition | undefined,
  state: PanelState,
  params: ParamRecord,
): boolean {
  if (!conditions) return true

  if (conditions.allConditionsTrue) {
    return conditions.allConditionsTrue.every((child) => evaluateCondition(child, state, params))
  }

  if (conditions.anyConditionsTrue) {
    return conditions.anyConditionsTrue.some((child) => evaluateCondition(child, state, params))
  }

  return evaluateCondition(conditions, state, params)
}

export function getDynamicDescription(
  control: PanelControl,
  state: PanelState,
  params: ParamRecord,
): string {
  if (!control.dynamic_desc || !control.descriptions || !control.description_conditions) {
    return control.desc || ''
  }

  for (const [key, condition] of Object.entries(control.description_conditions)) {
    if (evaluateConditions(condition, state, params)) {
      return control.descriptions[key] || control.desc || ''
    }
  }

  return control.descriptions.default || control.desc || ''
}

export function getDynamicTitle(
  control: PanelControl,
  _state: PanelState,
  params: ParamRecord,
): string {
  if (!control.dynamic_title || !control.titles) {
    return control.title || ''
  }

  if (control.type === 'toggle' && 'param' in control && control.param) {
    const value = getParamValue(params, control.param)
    const enabled = isParamTruthy(value)
    if (enabled && control.titles.enabled) return control.titles.enabled
    if (!enabled && control.titles.disabled) return control.titles.disabled
  }

  return control.title || ''
}

export function getDynamicStyle(
  control: PanelControl,
  _state: PanelState,
  params: ParamRecord,
): Record<string, string> {
  if (!control.dynamic_styling || !control.styles) {
    return control.button_style || {}
  }

  if (control.type === 'toggle' && 'param' in control && control.param) {
    const value = getParamValue(params, control.param)
    const enabled = isParamTruthy(value)
    if (enabled && control.styles.enabled) return control.styles.enabled
    if (!enabled && control.styles.disabled) return control.styles.disabled
  }

  return control.button_style || {}
}

export function isControlVisible(
  control: PanelControl,
  state: PanelState,
  params: ParamRecord,
): boolean {
  if (control.hidden === true) return false
  if (!control.visibleConditions) return true
  return evaluateConditions(control.visibleConditions, state, params)
}

export function isControlEnabled(
  control: PanelControl,
  state: PanelState,
  params: ParamRecord,
): boolean {
  if (!control.enableConditions) return true
  return evaluateConditions(control.enableConditions, state, params)
}

export function getDisabledReason(
  conditions: PanelCondition | undefined,
  state: PanelState,
  params: ParamRecord,
): string | null {
  if (!conditions) return null

  if (conditions.allConditionsTrue) {
    for (const child of conditions.allConditionsTrue) {
      if (!evaluateCondition(child, state, params)) {
        return child.reason || 'Condition not met'
      }
    }
    return null
  }

  if (conditions.anyConditionsTrue) {
    if (!conditions.anyConditionsTrue.some((child) => evaluateCondition(child, state, params))) {
      for (const child of conditions.anyConditionsTrue) {
        if (child.reason) return child.reason
      }
      return 'None of the required conditions are met'
    }
    return null
  }

  return evaluateCondition(conditions, state, params)
    ? null
    : conditions.reason || 'Condition not met'
}
