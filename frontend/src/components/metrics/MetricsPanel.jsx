import { useEffect, useState } from 'react'
import { getSystemInfo } from '../../lib/api.js'
import styles from './MetricsPanel.module.css'

export default function MetricsPanel() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    getSystemInfo()
      .then((data) => alive && setInfo(data))
      .catch((err) => alive && setError(err.message))
    return () => {
      alive = false
    }
  }, [])

  if (error) {
    return (
      <div className={styles.state}>
        <p className={styles.errorText}>Could not reach the backend API.</p>
        <code className={styles.code}>
          uvicorn backend.app:app --host 0.0.0.0 --port 8600 --workers 1
        </code>
        <p className={styles.errorDetail}>{error}</p>
      </div>
    )
  }

  if (!info) {
    return <div className={styles.state}>Loading system info…</div>
  }

  return (
    <div className={styles.panel}>
      <div className={styles.row}>
        <div className={styles.card}>
          <span className={styles.cardLabel}>GPU</span>
          <span className={styles.cardValue}>{info.gpu}</span>
        </div>
        <div className={styles.card}>
          <span className={styles.cardLabel}>CPU</span>
          <span className={styles.cardValue}>{info.cpu}</span>
        </div>
      </div>

      <div className={styles.versions}>
        <span className={styles.sectionLabel}>Runtime</span>
        <div className={styles.versionGrid}>
          {Object.entries(info.versions).map(([pkg, ver]) => (
            <div key={pkg} className={styles.versionChip}>
              <span className={styles.pkg}>{pkg}</span>
              <span className={styles.ver}>{ver}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
