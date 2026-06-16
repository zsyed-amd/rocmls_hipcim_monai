import styles from './Footer.module.css'

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
      <span>
        Powered by{' '}
        <a href="https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html" target="_blank" rel="noreferrer" className={styles.link}>AMD Instinct MI300X</a>
        {' '}&amp;{' '}
        <a href="https://rocm.docs.amd.com" target="_blank" rel="noreferrer" className={styles.link}>ROCm Open Software</a>
      </span>
      <span className={styles.year}>© {new Date().getFullYear()} Advanced Micro Devices, Inc. For demonstration purposes only.</span>
      </div>
    </footer>
  )
}
