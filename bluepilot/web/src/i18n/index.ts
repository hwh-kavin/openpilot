import { en, type TranslationKey } from './locales/en'
import { zhCHS } from './locales/zh-CHS'

export type PortalLocale = 'en' | 'zh-CHS'

const catalogs: Record<PortalLocale, Record<TranslationKey, string>> = {
  en,
  'zh-CHS': zhCHS,
}

export function resolvePortalLocale(languageSetting: string | null | undefined): PortalLocale {
  if (!languageSetting) {
    return 'en'
  }
  const lang = String(languageSetting).replace(/^main_/, '')
  if (lang.startsWith('zh')) {
    return 'zh-CHS'
  }
  return 'en'
}

export function translate(
  locale: PortalLocale,
  key: TranslationKey,
  vars?: Record<string, string | number>,
): string {
  let text = catalogs[locale][key] ?? catalogs.en[key] ?? key
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      text = text.split(`{${name}}`).join(String(value))
    }
  }
  return text
}

export { en, zhCHS }
export type { TranslationKey }
