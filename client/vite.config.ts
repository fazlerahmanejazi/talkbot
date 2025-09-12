import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/metrics': { target: 'http://localhost:8080', changeOrigin: true },
      '/health':  { target: 'http://localhost:8080', changeOrigin: true },
      '/ws':      { target: 'ws://localhost:8080', ws: true, changeOrigin: true }
    }
  }
})
