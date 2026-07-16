import { useLayoutEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useWebSocketStore } from '@/stores/useWebSocketStore'
import { useI18n } from '@/contexts/I18nContext'
import { Icon } from '@/components/common'
import type { DeviceStatus } from '@/types'
import './Header.css'

interface HeaderProps {
  deviceStatus?: DeviceStatus
  onMetricsClick?: () => void
  subtitle?: string
}

export const Header = ({
  deviceStatus = 'checking',
  onMetricsClick,
  subtitle,
}: HeaderProps = {}) => {
  const navigate = useNavigate()
  const location = useLocation()
  const { connected } = useWebSocketStore()
  const { t } = useI18n()
  const isHome = location.pathname === '/'
  const isRoutesPage = location.pathname.startsWith('/routes')
  const headerRef = useRef<HTMLElement | null>(null)

  const getTitle = () => {
    if (location.pathname === '/') return t('app.name')
    if (isRoutesPage) return t('nav.routes')
    if (location.pathname.startsWith('/parameters')) return t('nav.parameters')
    if (location.pathname.startsWith('/logs')) return t('nav.logs')
    if (location.pathname.startsWith('/settings')) return t('nav.settings')
    if (location.pathname.startsWith('/nav')) return t('nav.navigation')
    return t('app.name')
  }

  const isParametersPage = location.pathname.startsWith('/parameters')
  const isLogsPage = location.pathname.startsWith('/logs')

  const statusTexts: Record<DeviceStatus, string> = {
    online: t('status.online'),
    onroad: t('status.onroad'),
    offline: t('status.offline'),
    'no-network': t('status.noNetwork'),
    checking: t('status.checking'),
  }

  useLayoutEffect(() => {
    const updateHeaderHeight = () => {
      if (headerRef.current) {
        const height = headerRef.current.getBoundingClientRect().height
        document.documentElement.style.setProperty('--header-height', `${height}px`)
      }
    }

    updateHeaderHeight()
    window.addEventListener('resize', updateHeaderHeight)

    return () => {
      window.removeEventListener('resize', updateHeaderHeight)
    }
  }, [location.pathname, subtitle])

  return (
    <header className="header" ref={headerRef}>
      {!isHome && (
        <button
          className="icon-btn home-btn"
          onClick={() => navigate('/')}
          title={t('common.home')}
          type="button"
        >
          <Icon name="home" size={24} />
        </button>
      )}
      <div className="header-title-wrapper">
        <h1 className="header-title">{getTitle()}</h1>
        {subtitle && <p className="header-subtitle">{subtitle}</p>}
      </div>
      <div className="header-stats">
        <div className={`device-status ${deviceStatus} ${connected ? 'websocket-active' : ''}`} title={t('nav.deviceStatus')}>
          <span className="status-indicator"></span>
          <span id="status-text">{statusTexts[deviceStatus]}</span>
          {connected && (
            <Icon name="bolt" size={14} className="websocket-icon" />
          )}
        </div>
      </div>
      <div className="header-actions">
        {isLogsPage && (
          <button
            type="button"
            className="icon-btn"
            title={t('nav.parameters')}
            onClick={() => navigate('/parameters')}
          >
            <Icon name="tune" size={24} />
          </button>
        )}
        {isParametersPage && (
          <button
            type="button"
            className="icon-btn"
            title={t('nav.logs')}
            onClick={() => navigate('/logs')}
          >
            <Icon name="description" size={24} />
          </button>
        )}
        {isRoutesPage && (
          <button
            id="header-metrics-btn"
            type="button"
            className="icon-btn"
            title={t('nav.systemMetrics')}
            onClick={onMetricsClick}
          >
            <Icon name="bar_chart" size={24} />
          </button>
        )}
      </div>
    </header>
  )
}
