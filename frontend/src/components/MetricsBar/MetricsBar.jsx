import { useCountUp } from '../../lib/useCountUp.js'
import styles from './MetricsBar.module.css'

const METRICS = [
  { label: 'GPU Accelerated',     value: 192,  suffix: 'GB HBM3',  decimals: 0 },
  { label: 'Tile Throughput',     value: 512,  suffix: ' tiles/s',  decimals: 0 },
  { label: 'ROCm Compute Units',  value: 304,  suffix: ' CUs',      decimals: 0 },
  { label: 'Pathology Accuracy',  value: 97.3, suffix: '%',         decimals: 1 },
]

function MetricItem({ label, value, suffix, decimals }) {
  const animated = useCountUp(value, { duration: 1400 })
  const display = decimals > 0 ? animated.toFixed(decimals) : Math.round(animated)
  return (
    <div className={styles.metric}>
      <span className={styles.value}>{display}{suffix}</span>
      <span className={styles.label}>{label}</span>
    </div>
  )
}

export default function MetricsBar() {
  return (
    <div className={styles.bar}>
      <div className={styles.inner}>
        {METRICS.map((m) => (
          <MetricItem key={m.label} {...m} />
        ))}
      </div>
    </div>
  )
}
