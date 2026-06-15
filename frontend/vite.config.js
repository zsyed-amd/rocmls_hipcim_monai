import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8501',
      '/_stcore': 'http://localhost:8501',
      '/stream': 'http://localhost:8501',
      '/component': 'http://localhost:8501',
      '/healthz': 'http://localhost:8501',
    },
  },
})
