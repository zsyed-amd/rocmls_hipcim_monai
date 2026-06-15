import { useState, useEffect } from 'react'
import Header from './components/Header/Header.jsx'
import AppForm from './components/AppForm/AppForm.jsx'
import ResultsPanel from './components/ResultsPanel/ResultsPanel.jsx'
import MetricsBar from './components/MetricsBar/MetricsBar.jsx'
import FeatureCards from './components/FeatureCards/FeatureCards.jsx'
import OnboardingTour from './components/OnboardingTour/OnboardingTour.jsx'
import Footer from './components/Footer/Footer.jsx'
import styles from './App.module.css'

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('rocmls_theme') || 'dark')
  const [activeTab, setActiveTab] = useState('hipcim')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('rocmls_theme', theme)
  }, [theme])

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  return (
    <div className={styles.app}>
      <Header />
      <main className={styles.main}>
        <AppForm activeTab={activeTab} onTabChange={setActiveTab} />
        <ResultsPanel activeTab={activeTab} />
        <MetricsBar />
        <FeatureCards />
      </main>
      <Footer />
      <OnboardingTour theme={theme} onToggle={toggleTheme} />
    </div>
  )
}
