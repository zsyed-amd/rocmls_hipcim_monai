import { motion, AnimatePresence } from 'framer-motion'
import styles from './ResultsPanel.module.css'

const STREAMLIT_URL = 'http://localhost:8501'

const TAB_CONFIG = {
  hipcim: {
    label: 'hipCIM Viewer',
    anchor: '',
    caption: 'GPU-accelerated whole slide image processing — CPU vs AMD GPU tile comparison with LLaVA-NeXT pathology analysis.',
  },
  training: {
    label: 'MONAI Training',
    anchor: '',
    caption: 'Live training metrics: loss curves, Dice coefficient, accuracy, and sample visualizations per epoch.',
  },
  inference: {
    label: 'MONAI Inference',
    anchor: '',
    caption: 'Pathology tumor detection heatmaps and 3D spleen CT segmentation visualizations.',
  },
  metrics: {
    label: 'Performance Metrics',
    anchor: '',
    caption: 'Real-time throughput, GPU utilization, tile/sec, and batch latency on AMD Instinct MI300X.',
  },
}

export default function ResultsPanel({ activeTab }) {
  const config = TAB_CONFIG[activeTab] || TAB_CONFIG.hipcim

  return (
    <section className={styles.section} data-tour="results">
      <div className={styles.inner}>
        <div className={styles.panelHeader}>
          <div className={styles.panelTitle}>
            <span className={styles.goldAccent} aria-hidden="true" />
            {config.label}
          </div>
          <span className={styles.caption}>{config.caption}</span>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            className={styles.frameWrapper}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <iframe
              src={STREAMLIT_URL}
              title={`AMD ROCm Life Sciences Demo — ${config.label}`}
              className={styles.frame}
              allow="camera; microphone; display-capture"
            />
            <div className={styles.fallback}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={styles.fallbackIcon}>
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <p className={styles.fallbackText}>
                Start the backend to view this panel:
              </p>
              <code className={styles.fallbackCode}>streamlit run rocm-ls_demo.py</code>
              <a href={STREAMLIT_URL} target="_blank" rel="noreferrer" className={styles.fallbackLink}>
                Open in new tab →
              </a>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  )
}
