import styles from './Footer.module.css'

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <span>Powered by AMD Instinct MI300X &amp; ROCm Open Software</span>
      <span className={styles.year}>© {new Date().getFullYear()} Advanced Micro Devices, Inc.</span>
    </footer>
  )
}
