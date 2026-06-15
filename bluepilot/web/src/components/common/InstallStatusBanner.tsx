import { useEffect, useState } from 'react'
import { systemAPI } from '@/services/api'
import { useI18n } from '@/contexts/I18nContext'
import { Icon } from './Icon'
import './InstallStatusBanner.css'

interface InstallInfo {
  active: boolean
  phase?: string
  status?: string
  message?: string
  progress?: number | null
}

export const InstallStatusBanner = () => {
  const { t } = useI18n()
  const [install, setInstall] = useState<InstallInfo | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    let mounted = true

    const poll = async () => {
      try {
        const status = await systemAPI.getStatus()
        if (!mounted) return
        const info = status.install
        if (info?.active) {
          setInstall(info)
          setDismissed(false)
        } else if (info?.status === 'failed' && !dismissed) {
          setInstall(info)
        } else {
          setInstall(null)
        }
      } catch {
        if (mounted) setInstall(null)
      }
    }

    poll()
    const interval = setInterval(poll, 3000)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [dismissed])

  if (!install?.active && !(install?.status === 'failed' && !dismissed)) {
    return null
  }

  const isFailed = install.status === 'failed'
  const title = isFailed
    ? t('install.failed.title')
    : install.phase === 'bootstrap'
      ? t('install.bootstrap.title')
      : t('install.portal.title')

  const message = install.message || t('install.defaultMessage')
  const progress = typeof install.progress === 'number' ? install.progress : null

  return (
    <div className={`install-status-banner ${isFailed ? 'install-failed' : 'install-active'}`}>
      <div className="install-status-content">
        <Icon name={isFailed ? 'error' : 'download'} size={24} className="install-status-icon" />
        <div className="install-status-text">
          <strong>{title}</strong>
          <span>{message}</span>
          {!isFailed && progress !== null && (
            <div className="install-progress-track" aria-hidden="true">
              <div className="install-progress-fill" style={{ width: `${progress}%` }} />
            </div>
          )}
          {!isFailed && (
            <span className="install-status-hint">{t('install.keepPoweredOn')}</span>
          )}
        </div>
        {isFailed && (
          <button
            className="install-status-close"
            onClick={() => setDismissed(true)}
            title={t('common.dismiss')}
            type="button"
          >
            <Icon name="close" size={20} />
          </button>
        )}
      </div>
    </div>
  )
}
