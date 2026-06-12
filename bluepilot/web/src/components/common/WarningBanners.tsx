import { useState, useEffect } from 'react'
import { useI18n } from '@/contexts/I18nContext'
import { Icon } from './Icon'
import './WarningBanners.css'

export const WarningBanners = () => {
  const { t } = useI18n()
  const [showCellular, setShowCellular] = useState(false)
  const [showFirefox, setShowFirefox] = useState(false)

  useEffect(() => {
    // Check if Firefox
    const isFirefox = navigator.userAgent.toLowerCase().includes('firefox')
    if (isFirefox) {
      setShowFirefox(true)
    }

    // Check for cellular connection (placeholder - would need backend support)
    // setShowCellular(checkCellularStatus())
  }, [])

  if (!showCellular && !showFirefox) {
    return null
  }

  return (
    <>
      {showCellular && (
        <div className="cellular-warning">
          <div className="cellular-warning-content">
            <Icon name="warning" size={24} />
            <div className="cellular-warning-text">
              <strong>{t('warning.cellular.title')}</strong>
              <span>{t('warning.cellular.message')}</span>
            </div>
            <button className="cellular-warning-close" onClick={() => setShowCellular(false)} title={t('common.dismiss')}>
              <Icon name="close" size={20} />
            </button>
          </div>
        </div>
      )}

      {showFirefox && (
        <div className="firefox-warning">
          <div className="firefox-warning-content">
            <Icon name="info" size={24} />
            <div className="firefox-warning-text">
              <strong>{t('warning.firefox.title')}</strong>
              <span>{t('warning.firefox.message')}</span>
            </div>
            <button className="firefox-warning-close" onClick={() => setShowFirefox(false)} title={t('common.dismiss')}>
              <Icon name="close" size={20} />
            </button>
          </div>
        </div>
      )}
    </>
  )
}
