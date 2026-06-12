import { useEffect, useState, useRef, useCallback } from 'react'
import { Button, ToggleSwitch } from '@/components/common'

interface LogResponse {
  success: boolean
  output?: string
  error?: string
}

const POLL_INTERVAL_MS = 2000
const MAX_LINES = 2000

export function DiagnosticsTmux() {
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
          setError('Failed to fetch logs')
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
  }, [applyLogs, fetchLogs])

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

  useEffect(() => {
    if (autoScroll && logContainerRef.current && !isPaused) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logLines, autoScroll, isPaused])

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

  const handleRefresh = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const lines = await fetchLogs()
      applyLogs(lines)
    } catch (err) {
      setError('Failed to fetch logs')
    } finally {
      setIsLoading(false)
    }
  }

  const filteredLines = searchQuery
    ? logLines.filter(line => line.toLowerCase().includes(searchQuery.toLowerCase()))
    : logLines

  const displayText = filteredLines.join('\n')
  const streamLabel = isPaused ? 'Paused' : isLoading ? 'Loading...' : 'Streaming'

  return (
    <>
      <div className="diagnostics-controls">
        <div className="search-container">
          <input
            type="text"
            className="search-input"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              className="search-clear"
              onClick={() => setSearchQuery('')}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        <div className="control-buttons">
          <Button
            variant={isPaused ? 'primary' : 'secondary'}
            size="small"
            onClick={handlePauseToggle}
            icon={<span aria-hidden="true">{isPaused ? '▶' : '⏸'}</span>}
          >
            {isPaused ? 'Resume' : 'Pause'}
          </Button>

          <Button
            variant="secondary"
            size="small"
            onClick={handleClear}
            icon={<span aria-hidden="true">🗑</span>}
          >
            Clear
          </Button>

          <Button
            variant="primary"
            size="small"
            onClick={handleRefresh}
            disabled={isLoading}
            icon={<span aria-hidden="true">↻</span>}
          >
            Refresh
          </Button>

          <ToggleSwitch
            checked={autoScroll}
            onChange={setAutoScroll}
            label="Auto-scroll"
            size="compact"
            className="diagnostics-toggle"
          />
        </div>
      </div>

      <div className="diagnostics-content console-content">
        {error && (
          <div className="console-error">
            {error}
          </div>
        )}

        <div className="console-status">
          <span className="console-status-label">Status:</span>
          <span className={`console-status-value ${!isPaused && !isLoading ? 'streaming' : 'paused'}`}>
            {streamLabel}
          </span>
          {searchQuery && (
            <>
              <span className="console-status-separator">•</span>
              <span className="console-status-label">Showing:</span>
              <span className="console-status-value">
                {filteredLines.length} / {logLines.length} lines
              </span>
            </>
          )}
          <span className="console-status-separator">•</span>
          <span className="console-status-label">Total:</span>
          <span className="console-status-value">{logLines.length} lines</span>
        </div>

        <pre
          ref={logContainerRef}
          className="console-output"
          onScroll={handleScroll}
          aria-live={isPaused ? 'off' : 'polite'}
        >
          {displayText || (isLoading ? 'Loading logs...' : 'No logs available')}
        </pre>
      </div>
    </>
  )
}
