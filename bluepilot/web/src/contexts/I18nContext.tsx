import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { systemAPI } from '@/services/api'
import { useWebSocketStore } from '@/stores/useWebSocketStore'
import { PortalLocale, TranslationKey, resolvePortalLocale, translate } from '@/i18n'

interface I18nContextValue {
  locale: PortalLocale
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string
  setLocale: (locale: PortalLocale) => void
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<PortalLocale>('en')
  const lastParamUpdate = useWebSocketStore((state) => state.lastParamUpdate)

  const refreshLocale = useCallback(async () => {
    try {
      const status = await systemAPI.getStatus()
      if (status.language) {
        setLocale(resolvePortalLocale(status.language))
      }
    } catch {
      // Keep current locale when status is unavailable.
    }
  }, [])

  useEffect(() => {
    refreshLocale()
  }, [refreshLocale])

  useEffect(() => {
    if (lastParamUpdate?.key === 'LanguageSetting') {
      setLocale(resolvePortalLocale(String(lastParamUpdate.value)))
    }
  }, [lastParamUpdate])

  useEffect(() => {
    document.documentElement.lang = locale === 'zh-CHS' ? 'zh-CN' : 'en'
  }, [locale])

  const t = useCallback(
    (key: TranslationKey, vars?: Record<string, string | number>) => translate(locale, key, vars),
    [locale],
  )

  const value = useMemo(() => ({ locale, t, setLocale }), [locale, t])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const context = useContext(I18nContext)
  if (!context) {
    throw new Error('useI18n must be used within I18nProvider')
  }
  return context
}
