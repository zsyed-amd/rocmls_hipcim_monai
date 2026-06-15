import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './FeatureCards.module.css'

const FEATURES = [
  {
    eyebrow: 'Hardware',
    title: 'AMD Instinct MI300X',
    description: '192GB HBM3 unified memory enables processing of gigapixel whole slide images and large 3D CT volumes without data paging.',
    detail: 'The AMD Instinct MI300X delivers 192GB of HBM3 memory with 5.3 TB/s bandwidth — purpose-built for memory-intensive medical imaging workloads. Its unified CPU+GPU memory architecture eliminates data transfer bottlenecks when processing multi-gigabyte pathology slides and 3D NIfTI volumes. With 304 compute units and ROCm open-source support, it runs the full MONAI and hipCIM stack natively.',
    links: [
      { label: 'MI300X product page', href: 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html' },
      { label: 'ROCm documentation',  href: 'https://rocm.docs.amd.com' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" />
        <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
        <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
        <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" />
        <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" />
      </svg>
    ),
  },
  {
    eyebrow: 'Software',
    title: 'ROCm · MONAI · hipCIM',
    description: 'AMD\'s open-source ROCm stack powers MONAI medical AI training and hipCIM GPU-accelerated pathology image processing.',
    detail: 'ROCm provides the open-source GPU compute foundation: HIP for GPU kernels, rocBLAS for linear algebra, and MIOpen for deep learning primitives. MONAI (Medical Open Network for AI) delivers pre-built training pipelines, model zoo bundles, and transforms for 3D medical images. hipCIM, AMD\'s CuCIM replacement, accelerates whole slide image tiling, stain separation, and Gabor filtering directly on the GPU — replacing CPU-bound OpenSlide with hardware-accelerated alternatives.',
    links: [
      { label: 'ROCm documentation', href: 'https://rocm.docs.amd.com' },
      { label: 'MONAI project',       href: 'https://monai.io' },
      { label: 'ROCm on GitHub',      href: 'https://github.com/ROCm/ROCm' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2" />
        <polyline points="2 17 12 22 22 17" />
        <polyline points="2 12 12 17 22 12" />
      </svg>
    ),
  },
  {
    eyebrow: 'Performance',
    title: 'GPU-Accelerated Throughput',
    description: 'AMD MI300X processes 512+ pathology tiles per second — orders of magnitude faster than CPU-bound OpenSlide pipelines.',
    detail: 'hipCIM\'s GPU-accelerated tile extraction and batch inference pipeline achieves 512+ tiles/second on MI300X, compared to single-digit throughput on CPU-only workflows. MONAI training with DataParallel multi-GPU support reduces epoch times dramatically on large datasets. The DataLoader pipeline uses 16 parallel workers with GPU-batched inference at batch_size=512, saturating HBM3 bandwidth for maximum throughput on gigapixel WSI analysis.',
    links: [
      { label: 'AMD Instinct performance', href: 'https://www.amd.com/en/products/accelerators/instinct.html' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="13 2 13 9 19 9" />
        <path d="M19 2H13L3 12h7l-2 10 13-13h-7l5-9z" />
      </svg>
    ),
  },
  {
    eyebrow: 'Capability',
    title: 'Pathology AI & CT Segmentation',
    description: 'LLaVA-NeXT vision-language analysis, tumor detection on gigapixel slides, and 3D spleen CT segmentation — all on one GPU.',
    detail: 'The demo showcases two MONAI model bundles: a ResNet18-based pathology tumor detection model that classifies tiles from Camelyon16-format whole slide images, and a 3D UNet for spleen CT segmentation from NIfTI volumes. Results include annotated heatmaps with red bounding boxes on detected tumor regions, per-tile CSV exports, and 3D slice visualizations. LLaVA-NeXT provides natural-language pathology interpretation of selected tiles, making findings accessible beyond raw model outputs.',
    links: [
      { label: 'MONAI Model Zoo',   href: 'https://monai.io/model-zoo.html' },
      { label: 'Camelyon16 dataset', href: 'https://camelyon16.grand-challenge.org' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6"  y1="20" x2="6"  y2="14" />
        <line x1="2"  y1="20" x2="22" y2="20" />
      </svg>
    ),
  },
]

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
}

const cardVariants = {
  hidden:  { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
}

export default function FeatureCards() {
  const [active, setActive] = useState(null)
  const feature = FEATURES.find((f) => f.title === active)

  return (
    <>
      <AnimatePresence>
        {active && (
          <motion.div
            className={styles.overlay}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={() => setActive(null)}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {active && feature && (
          <motion.aside
            className={styles.panel}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className={styles.panelHeader}>
              <div className={styles.panelHeaderLeft}>
                <div className={styles.panelIconBox}>{feature.icon}</div>
                <div>
                  <div className={styles.panelEyebrow}>{feature.eyebrow}</div>
                  <div className={styles.panelName}>{feature.title}</div>
                </div>
              </div>
              <button className={styles.panelClose} onClick={() => setActive(null)} aria-label="Close panel">
                &times;
              </button>
            </div>
            <div className={styles.panelBody}>
              <p className={styles.panelDetail}>{feature.detail}</p>
              {feature.links.length > 0 && (
                <div className={styles.panelLinks}>
                  <span className={styles.panelLinksLabel}>Learn more</span>
                  {feature.links.map((l) => (
                    <a key={l.href} href={l.href} target="_blank" rel="noreferrer" className={styles.panelLink}>
                      {l.label}
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M7 17L17 7M17 7H7M17 7v10" />
                      </svg>
                    </a>
                  ))}
                </div>
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <motion.section
        className={styles.section}
        data-tour="cards"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {FEATURES.map((f) => (
          <motion.div
            key={f.title}
            className={`${styles.card} ${active === f.title ? styles.cardActive : ''}`}
            variants={cardVariants}
            onClick={() => setActive((prev) => (prev === f.title ? null : f.title))}
            role="button"
            aria-expanded={active === f.title}
            aria-label={`${f.title} — ${f.eyebrow}. Click to ${active === f.title ? 'close' : 'learn more'}`}
          >
            <div className={styles.header}>
              <span className={styles.accent} aria-hidden="true" />
              <span className={styles.icon} aria-hidden="true">{f.icon}</span>
            </div>
            <h3 className={styles.title}>{f.title}</h3>
            <p className={styles.description}>{f.description}</p>
          </motion.div>
        ))}
      </motion.section>
    </>
  )
}
