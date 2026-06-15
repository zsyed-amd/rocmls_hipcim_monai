import styles from './AMDLoader.module.css'

export default function AMDLoader({ label = 'Processing with AMD MI300X...' }) {
  return (
    <div className={styles.wrapper}>
      <svg width="48" height="48" viewBox="0 0 48 48" className={styles.svg}>
        <defs>
          <linearGradient id="shimmer" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#00C2DE" />
            <stop offset="50%"  stopColor="#C1A968" />
            <stop offset="100%" stopColor="#00C2DE" />
            <animateTransform
              attributeName="gradientTransform"
              type="translate"
              from="-1 0"
              to="1 0"
              dur="1.6s"
              repeatCount="indefinite"
            />
          </linearGradient>
        </defs>
        <path
          d="M8 40 L24 8 L40 40"
          stroke="url(#shimmer)"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
        <path
          d="M14 30 L34 30"
          stroke="url(#shimmer)"
          strokeWidth="2.5"
          strokeLinecap="round"
          fill="none"
        />
      </svg>
      <span className={styles.label}>{label}</span>
      <div className={styles.dots}>
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
    </div>
  )
}
