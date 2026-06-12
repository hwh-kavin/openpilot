import { useEffect, useState, useRef, useMemo, useCallback } from 'react'
import { Header } from '@/components/layout/Header'
import { Button, ToggleSwitch, Icon, BackToTop } from '@/components/common'
import { useI18n } from '@/contexts/I18nContext'
import type { DeviceStatus } from '@/types'
import './LogsView.css'

interface LogResponse {
  success: boolean
  output?: string
  error?: string
}

interface LogsViewProps {
  deviceStatus?: DeviceStatus
}

const POLL_INTERVAL_MS = 2000
const MAX_LINES = 2000

// ANSI color code parser
const parseAnsiColors = (text: string): JSX.Element[] => {
  const ansiRegex = /\u001b\[([0-9;]+)m/g
  const parts: JSX.Element[] = []
  let lastIndex = 0
  let currentClasses: string[] = []
  let keyCounter = 0

  const ansiCodeToClass = (code: number): string | null => {
    // Foreground colors
    if (code === 30) return 'ansi-black'
    if (code === 31) return 'ansi-red'
    if (code === 32) return 'ansi-green'
    if (code === 33) return 'ansi-yellow'
    if (code === 34) return 'ansi-blue'
    if (code === 35) return 'ansi-magenta'
    if (code === 36) return 'ansi-cyan'
    if (code === 37) return 'ansi-white'

    // Bright foreground colors
    if (code === 90) return 'ansi-bright-black'
    if (code === 91) return 'ansi-bright-red'
    if (code === 92) return 'ansi-bright-green'
    if (code === 93) return 'ansi-bright-yellow'
    if (code === 94) return 'ansi-bright-blue'
    if (code === 95) return 'ansi-bright-magenta'
    if (code === 96) return 'ansi-bright-cyan'
    if (code === 97) return 'ansi-bright-white'

    // Background colors
    if (code === 40) return 'ansi-bg-black'
    if (code === 41) return 'ansi-bg-red'
    if (code === 42) return 'ansi-bg-green'
    if (code === 43) return 'ansi-bg-yellow'
    if (code === 44) return 'ansi-bg-blue'
    if (code === 45) return 'ansi-bg-magenta'
    if (code === 46) return 'ansi-bg-cyan'
    if (code === 47) return 'ansi-bg-white'

    // Text styles
    if (code === 1) return 'ansi-bold'
    if (code === 2) return 'ansi-dim'
    if (code === 3) return 'ansi-italic'
    if (code === 4) return 'ansi-underline'

    // Reset
    if (code === 0) return null

    return null
  }

  let match: RegExpExecArray | null
  while ((match = ansiRegex.exec(text)) !== null) {
    // Add text before this ANSI code
    if (match.index > lastIndex) {
      const textContent = text.substring(lastIndex, match.index)
      if (currentClasses.length > 0) {
        parts.push(
          <span key={`span-${keyCounter++}`} className={currentClasses.join(' ')}>
            {textContent}
          </span>
        )
      } else {
        parts.push(<span key={`span-${keyCounter++}`}>{textContent}</span>)
      }
    }

    // Parse ANSI codes
    const codes = match[1].split(';').map(Number)
    for (const code of codes) {
      if (code === 0) {
        currentClasses = []
      } else {
        const className = ansiCodeToClass(code)
        if (className) {
          currentClasses.push(className)
        }
      }
    }

    lastIndex = ansiRegex.lastIndex
  }

  // Add remaining text
  if (lastIndex < text.length) {
    const textContent = text.substring(lastIndex)
    if (currentClasses.length > 0) {
      parts.push(
        <span key={`span-${keyCounter++}`} className={currentClasses.join(' ')}>
          {textContent}
        </span>
      )
    } else {
      parts.push(<span key={`span-${keyCounter++}`}>{textContent}</span>)
    }
  }

  return parts.length > 0 ? parts : [<span key="span-0">{text}</span>]
}

export function LogsView({ deviceStatus = 'checking' }: LogsViewProps) {
  const { t } = useI18n()
  const [logLines, setLogLines] = useState<string[]>([])
  const [isPaused, setIsPaused] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  const logContainerRef = useRef<HTMLPreElement>(null)

  const fetchLogs = useCallback(async (): Promise<string[]> => {
    const response = await fetch('/api/manager-logs')
    if (!response.ok) {
      throw new Error('Failed to fetch logs')
    }

    const data: LogResponse = await response.json()
    if (!data.success || !data.output) {
      throw new Error(data.error || 'No log output')
    }

    return data.output.split('\n').filter(line => line.trim())
  }, [])

  const applyLogs = useCallback((lines: string[]) => {
    setLogLines(lines.slice(-MAX_LINES))
    setError(null)
  }, [])

  // Initial load
  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const lines = await fetchLogs()
        if (!cancelled) {
          applyLogs(lines)
        }
      } catch (err) {
        if (!cancelled) {
          console.error('Failed to fetch initial logs:', err)
          setError(t('logs.fetchError'))
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false)
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [applyLogs, fetchLogs, t])

  // Poll swaglog files while the page is active and not paused
  useEffect(() => {
    if (isPaused) {
      return
    }

    const poll = async () => {
      try {
        const lines = await fetchLogs()
        applyLogs(lines)
      } catch (err) {
        console.error('Failed to poll logs:', err)
      }
    }

    const timer = window.setInterval(poll, POLL_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [applyLogs, fetchLogs, isPaused])

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current && !isPaused) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logLines, autoScroll, isPaused])

  // Detect manual scrolling
  const handleScroll = () => {
    if (logContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50
      setAutoScroll(isAtBottom)
    }
  }

  const handlePauseToggle = () => {
    setIsPaused(prev => !prev)
  }

  const handleClear = () => {
    setLogLines([])
  }

  // Export logs to file
  const handleExport = useCallback(() => {
    if (logLines.length === 0) return

    // Strip ANSI codes for clean export
    const stripAnsi = (text: string) => text.replace(/\u001b\[[0-9;]+m/g, '')
    const cleanLogs = logLines.map(stripAnsi).join('\n')

    const blob = new Blob([cleanLogs], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `bluepilot-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [logLines])

  const handleRefresh = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const lines = await fetchLogs()
      applyLogs(lines)
    } catch (err) {
      setError(t('logs.fetchError'))
    } finally {
      setIsLoading(false)
    }
  }

  // Filter lines based on search query
  const filteredLines = searchQuery
    ? logLines.filter(line => line.toLowerCase().includes(searchQuery.toLowerCase()))
    : logLines

  // Parse ANSI colors for display
  const displayContent = useMemo(() => {
    return filteredLines.map((line, index) => (
      <div key={`line-${index}`}>
        {parseAnsiColors(line)}
      </div>
    ))
  }, [filteredLines])

  const streamLabel = isPaused
    ? t('status.paused')
    : isLoading
      ? t('logs.loading')
      : t('status.streaming')

  return (
    <>
      <Header deviceStatus={deviceStatus} subtitle={t('logs.subtitle')} />
      <div className="logs-view">
        <div className="logs-controls">
          <div className="search-container">
            <input
              type="text"
              className="search-input"
              placeholder={t('logs.search')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                className="search-clear"
                onClick={() => setSearchQuery('')}
                aria-label={t('logs.clearSearch')}
              >
                <Icon name="close" size={18} />
              </button>
            )}
          </div>

          <div className="control-buttons">
            <Button
              variant={isPaused ? 'primary' : 'secondary'}
              size="small"
              onClick={handlePauseToggle}
              icon={<Icon name={isPaused ? 'play_arrow' : 'pause'} size={18} />}
            >
              {isPaused ? t('common.resume') : t('common.pause')}
            </Button>

            <Button
              variant="secondary"
              size="small"
              onClick={handleClear}
              icon={<Icon name="delete" size={18} />}
            >
              {t('common.clear')}
            </Button>

            <Button
              variant="primary"
              size="small"
              onClick={handleRefresh}
              disabled={isLoading}
              icon={<Icon name="refresh" size={18} />}
            >
              {t('common.refresh')}
            </Button>

            <Button
              variant="secondary"
              size="small"
              onClick={handleExport}
              disabled={logLines.length === 0}
              icon={<Icon name="download" size={18} />}
            >
              {t('common.export')}
            </Button>

            <ToggleSwitch
              checked={autoScroll}
              onChange={setAutoScroll}
              label={t('logs.autoScroll')}
              size="compact"
              className="logs-toggle"
            />
          </div>
        </div>

        <div className="logs-content">
          {error && (
            <div className="console-error">
              {error}
            </div>
          )}

          <div className="console-status">
            <span className="console-status-label">{t('logs.status')}:</span>
            <span className={`console-status-value ${!isPaused && !isLoading ? 'streaming' : 'paused'}`}>
              {streamLabel}
            </span>
            {searchQuery && (
              <>
                <span className="console-status-separator">•</span>
                <span className="console-status-label">{t('logs.showing')}:</span>
                <span className="console-status-value">
                  {filteredLines.length} / {logLines.length} {t('logs.lines')}
                </span>
              </>
            )}
            <span className="console-status-separator">•</span>
            <span className="console-status-label">{t('logs.total')}:</span>
            <span className="console-status-value">{logLines.length} {t('logs.lines')}</span>
          </div>

          <pre
            ref={logContainerRef}
            className="console-output"
            onScroll={handleScroll}
            aria-live={isPaused ? 'off' : 'polite'}
          >
            {displayContent.length > 0 ? displayContent : (isLoading ? t('logs.loading') : t('logs.empty'))}
          </pre>
        </div>
      </div>

      <BackToTop />
    </>
  )
}
