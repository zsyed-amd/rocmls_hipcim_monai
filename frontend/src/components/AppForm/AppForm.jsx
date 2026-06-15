import { motion } from 'framer-motion'
import styles from './AppForm.module.css'

const TABS = [
  { id: 'hipcim',   label: 'hipCIM Viewer' },
  { id: 'training', label: 'MONAI Training' },
  { id: 'inference', label: 'MONAI Inference' },
  { id: 'metrics',  label: 'Performance Metrics' },
]

const TAB_DESCRIPTIONS = {
  hipcim:   'GPU-accelerated whole slide image processing with hipCIM. Upload a TIFF file and select image transformations to compare CPU vs AMD GPU performance side-by-side.',
  training: 'Train medical AI models on AMD Instinct MI300X. Select a MONAI model bundle (pathology tumor detection or spleen CT segmentation), configure hyperparameters, and watch live loss curves.',
  inference: 'Run inference with pre-trained or custom MONAI models. Upload NIfTI CT volumes or pathology TIFF slides and get AI-powered segmentation and classification results.',
  metrics:  'Monitor real-time GPU utilization, tile throughput, and batch latency across hipCIM and MONAI workloads running on AMD Instinct MI300X.',
}

export default function AppForm({ activeTab, onTabChange }) {
  return (
    <section className={styles.section} data-tour="form">
      <div className={styles.inner}>
        <div className={styles.tabs} role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={activeTab === t.id}
              className={`${styles.tab} ${activeTab === t.id ? styles.tabActive : ''}`}
              onClick={() => onTabChange(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <motion.div
          key={activeTab}
          className={styles.body}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.22, ease: 'easeOut' }}
        >
          <div className={styles.intro}>
            <h2 className={styles.heading}>{TABS.find(t => t.id === activeTab)?.label}</h2>
            <p className={styles.description}>{TAB_DESCRIPTIONS[activeTab]}</p>
          </div>

          <div className={styles.launchWrapper} data-tour="submit">
            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noreferrer"
              className={styles.launchBtn}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
              Launch Full Demo App
            </a>
            <span className={styles.launchNote}>Opens the Streamlit interface on localhost:8501</span>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
