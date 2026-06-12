import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Header } from '@/components/layout/Header'
import { BackToTop, Button, Icon, LoadingSpinner, Modal } from '@/components/common'
import { FavoritesPanel } from '@/components/settings/FavoritesPanel'
import { FloatingChangesIndicator } from '@/components/settings/FloatingChangesIndicator'
import { PanelGroup } from '@/components/settings/PanelGroup'
import { useUnsavedChangesWarning } from '@/hooks/useUnsavedChangesWarning'
import { useParamsStore } from '@/stores/useParamsStore'
import { usePanelStateStore } from '@/stores/usePanelStateStore'
import { usePanelsStore } from '@/stores/usePanelsStore'
import type { DeviceStatus } from '@/types'
import type { PanelMetadata } from '@/types/panels'
import './SettingsView.css'

interface SettingsViewProps {
  deviceStatus?: DeviceStatus
}

const FAVORITES_PANEL: PanelMetadata = {
  id: 'favorites',
  name: 'Favorites',
  description: 'Pinned controls from every panel for quick access',
}

const PANEL_ICONS: Record<string, string> = {
  favorites: 'star',
  bp_device_panel: 'devices',
  bp_display_panel: 'monitor',
  bp_visuals_panel: 'visibility',
  bp_vehicle_panel: 'directions_car',
  bp_cruise_panel: 'speed',
  bp_toggles_panel: 'toggle_on',
  bp_steering_panel: 'trip_origin',
  bp_developer_panel: 'code',
}

function renderPanelIcon(panelId?: string) {
  const iconName = (panelId && PANEL_ICONS[panelId]) || 'dashboard'
  return <Icon name={iconName} />
}

export const SettingsView = ({ deviceStatus = 'checking' }: SettingsViewProps) => {
  const { panels, loadedPanels, loading, error, fetchPanels, fetchPanel } = usePanelsStore()
  const { state, fetchState } = usePanelStateStore()
  const { fetchParams } = useParamsStore()

  const [activePanelId, setActivePanelId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)
  const [backupResult, setBackupResult] = useState<{
    success: boolean
    message: string
    details?: { restored?: number; failed?: number }
  } | null>(null)
  const [showBackupResult, setShowBackupResult] = useState(false)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const mobileNavRef = useRef<HTMLDivElement>(null)

  useUnsavedChangesWarning()

  const selectPanel = useCallback((panelId: string) => {
    setActivePanelId(panelId)
    setMobileNavOpen(false)
  }, [])

  useEffect(() => {
    fetchPanels()
    fetchState()
    fetchParams()
  }, [fetchPanels, fetchState, fetchParams])

  useEffect(() => {
    if (!activePanelId) {
      setActivePanelId('favorites')
    }
  }, [activePanelId])

  useEffect(() => {
    if (activePanelId && activePanelId !== 'favorites' && !loadedPanels[activePanelId]) {
      fetchPanel(activePanelId)
    }
  }, [activePanelId, loadedPanels, fetchPanel])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (mobileNavRef.current && !mobileNavRef.current.contains(event.target as Node)) {
        setMobileNavOpen(false)
      }
    }

    if (mobileNavOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [mobileNavOpen])

  const handleExport = async () => {
    setExporting(true)
    setBackupResult(null)

    try {
      const response = await fetch('/api/params/backup')
      const data = await response.json()

      if (data.success) {
        const backup = {
          version: '1.0',
          timestamp: new Date().toISOString(),
          device: 'BluePilot',
          params: data.params,
          count: data.count,
        }

        const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `bluepilot-backup-${new Date().toISOString().split('T')[0]}.json`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)

        setBackupResult({ success: true, message: `Exported ${data.count} settings` })
        setShowBackupResult(true)
      } else {
        setBackupResult({ success: false, message: data.error || 'Export failed' })
        setShowBackupResult(true)
      }
    } catch (err) {
      setBackupResult({
        success: false,
        message: err instanceof Error ? err.message : 'Export failed',
      })
      setShowBackupResult(true)
    } finally {
      setExporting(false)
    }
  }

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setImporting(true)
    setBackupResult(null)

    try {
      const text = await file.text()
      const backup = JSON.parse(text)

      if (!backup.params || !backup.version) {
        throw new Error('Invalid backup file format')
      }

      const response = await fetch('/api/params/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: backup.params }),
      })

      const data = await response.json()

      if (data.success || (data.restored?.length ?? 0) > 0) {
        setBackupResult({
          success: true,
          message: `Restored ${data.count || data.restored?.length || 0} settings`,
          details: {
            restored: data.restored?.length || 0,
            failed: data.failed?.length || 0,
          },
        })
        fetchParams()
      } else {
        setBackupResult({ success: false, message: data.error || 'Restore failed' })
      }
      setShowBackupResult(true)
    } catch (err) {
      setBackupResult({
        success: false,
        message: err instanceof Error ? err.message : 'Import failed',
      })
      setShowBackupResult(true)
    } finally {
      setImporting(false)
      event.target.value = ''
    }
  }

  const activePanelConfig = activePanelId && activePanelId !== 'favorites'
    ? loadedPanels[activePanelId]
    : null

  const filteredGroups = useMemo(() => {
    if (!activePanelConfig) return undefined

    const query = searchQuery.trim().toLowerCase()
    if (!query) return activePanelConfig.groups

    return activePanelConfig.groups
      .map((group) => ({
        ...group,
        controls: group.controls.filter((control) => {
          if ('webSupported' in control && control.webSupported === false) return false
          const title = control.title?.toLowerCase() || ''
          const desc = control.desc?.toLowerCase() || ''
          return title.includes(query) || desc.includes(query)
        }),
      }))
      .filter((group) => group.controls.length > 0)
  }, [activePanelConfig, searchQuery])

  const activePanelMeta = activePanelId === 'favorites'
    ? FAVORITES_PANEL
    : panels.find((panel) => panel.id === activePanelId) || null

  const panelDescription = activePanelId === 'favorites'
    ? FAVORITES_PANEL.description
    : activePanelConfig?.menuDescription || activePanelMeta?.description || 'Configure BluePilot behavior offroad'

  const renderNavItem = (panel: PanelMetadata, isActive: boolean, onClick: () => void, className: string) => (
    <button
      key={panel.id}
      type="button"
      className={`${className} ${isActive ? 'active' : ''}`}
      onClick={onClick}
      data-panel-id={panel.id}
    >
      <div className="settings-nav-icon">{renderPanelIcon(panel.id)}</div>
      <div className="settings-nav-copy">
        <span className="settings-nav-label">{panel.name}</span>
        <span className="settings-nav-desc">{panel.description || 'Panel controls'}</span>
      </div>
    </button>
  )

  if (loading && panels.length === 0) {
    return (
      <>
        <Header deviceStatus={deviceStatus} subtitle="Configure BluePilot settings and behavior" />
        <div className="settings-view settings-view-centered">
          <LoadingSpinner message="Loading settings..." />
        </div>
      </>
    )
  }

  if (error) {
    return (
      <>
        <Header deviceStatus={deviceStatus} subtitle="Configure BluePilot settings and behavior" />
        <div className="settings-view settings-view-centered">
          <div className="settings-error-card">
            <h2>Error Loading Settings</h2>
            <p>{error}</p>
          </div>
        </div>
      </>
    )
  }

  const selectedMobilePanel = activePanelId === 'favorites'
    ? FAVORITES_PANEL
    : panels.find((panel) => panel.id === activePanelId)

  return (
    <>
      <Header deviceStatus={deviceStatus} subtitle="Configure BluePilot settings and behavior" />

      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleImport}
        disabled={importing}
        className="settings-file-input-hidden"
        aria-label="Import settings file"
      />

      <div className="settings-view">
        <div className="settings-layout">
          <aside className="settings-sidebar">
            <div className="settings-nav-mobile" ref={mobileNavRef}>
              <div className="settings-backup-actions">
                <button
                  type="button"
                  className="settings-backup-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing}
                  title="Import settings"
                >
                  <Icon name="upload" size={18} />
                  <span>{importing ? '...' : 'Import'}</span>
                </button>
                <button
                  type="button"
                  className="settings-backup-btn"
                  onClick={handleExport}
                  disabled={exporting}
                  title="Export settings"
                >
                  <Icon name="download" size={18} />
                  <span>{exporting ? '...' : 'Export'}</span>
                </button>
              </div>

              <button
                type="button"
                className={`settings-nav-dropdown-trigger ${mobileNavOpen ? 'open' : ''}`}
                onClick={() => setMobileNavOpen((open) => !open)}
                aria-label="Select settings panel"
              >
                <div className="settings-nav-dropdown-selected">
                  <div className="settings-nav-icon">{renderPanelIcon(activePanelId || 'favorites')}</div>
                  <div className="settings-nav-copy">
                    <span className="settings-nav-label">
                      {activePanelId === 'favorites' ? 'Favorites' : selectedMobilePanel?.name || 'Select Panel'}
                    </span>
                    <span className="settings-nav-desc">
                      {activePanelId === 'favorites' ? 'Starred controls' : selectedMobilePanel?.description || ''}
                    </span>
                  </div>
                </div>
                <Icon name="expand_more" size={20} className="settings-nav-dropdown-chevron" />
              </button>

              {mobileNavOpen && (
                <div className="settings-nav-dropdown-menu">
                  {renderNavItem(FAVORITES_PANEL, activePanelId === 'favorites', () => selectPanel('favorites'), 'settings-nav-dropdown-item')}
                  {panels.map((panel) =>
                    renderNavItem(panel, activePanelId === panel.id, () => selectPanel(panel.id), 'settings-nav-dropdown-item'),
                  )}
                </div>
              )}
            </div>

            <div className="settings-nav">
              <div className="settings-backup-actions">
                <button
                  type="button"
                  className="settings-backup-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing}
                  title="Import settings"
                >
                  <Icon name="upload" size={18} />
                  <span>{importing ? '...' : 'Import'}</span>
                </button>
                <button
                  type="button"
                  className="settings-backup-btn"
                  onClick={handleExport}
                  disabled={exporting}
                  title="Export settings"
                >
                  <Icon name="download" size={18} />
                  <span>{exporting ? '...' : 'Export'}</span>
                </button>
              </div>

              {renderNavItem(FAVORITES_PANEL, activePanelId === 'favorites', () => selectPanel('favorites'), 'settings-nav-btn')}
              {panels.map((panel) =>
                renderNavItem(panel, activePanelId === panel.id, () => selectPanel(panel.id), 'settings-nav-btn'),
              )}
            </div>
          </aside>

          <section className="settings-main">
            <div className="settings-panel-header">
              <div className="settings-panel-heading">
                <div className="settings-panel-icon">{renderPanelIcon(activePanelMeta?.id)}</div>
                <div>
                  <h1>{activePanelMeta?.name || 'Settings'}</h1>
                  <p>{panelDescription}</p>
                </div>
              </div>

              <div className="settings-search">
                <input
                  type="text"
                  className="settings-search-input"
                  placeholder="Search settings..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  aria-label="Search settings"
                />
                {searchQuery && (
                  <button
                    type="button"
                    className="settings-search-clear"
                    onClick={() => setSearchQuery('')}
                    aria-label="Clear search"
                  >
                    <Icon name="close" size={18} />
                  </button>
                )}
              </div>
            </div>

            <div className="settings-panel-content">
              {activePanelId === 'favorites' ? (
                <FavoritesPanel />
              ) : activePanelConfig ? (
                <>
                  {!searchQuery && panelDescription && (
                    <div className="settings-panel-description">{panelDescription}</div>
                  )}
                  {filteredGroups && filteredGroups.length > 0 ? (
                    filteredGroups.map((group) => (
                      <PanelGroup
                        key={group.groupName}
                        group={group}
                        state={state}
                        panelId={activePanelId || undefined}
                      />
                    ))
                  ) : searchQuery ? (
                    <div className="settings-no-results">
                      <p>{`No settings found for "${searchQuery}"`}</p>
                      <button type="button" onClick={() => setSearchQuery('')}>
                        Clear search
                      </button>
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="settings-panel-loading">
                  <LoadingSpinner message="Loading panel..." />
                </div>
              )}
            </div>
          </section>
        </div>
      </div>

      {showBackupResult && backupResult && (
        <Modal
          isOpen={showBackupResult}
          title={backupResult.success ? 'Success' : 'Error'}
          onClose={() => setShowBackupResult(false)}
        >
          <p>{backupResult.message}</p>
          {backupResult.details && (backupResult.details.failed ?? 0) > 0 && (
            <p className="settings-backup-detail">
              <strong>Failed:</strong> {backupResult.details.failed}
            </p>
          )}
          <div className="settings-backup-modal-actions">
            <Button variant="primary" onClick={() => setShowBackupResult(false)}>
              OK
            </Button>
          </div>
        </Modal>
      )}

      <FloatingChangesIndicator />
      <BackToTop />
    </>
  )
}
