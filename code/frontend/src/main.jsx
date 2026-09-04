import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

createRoot(document.getElementById('root')).render(
  // StrictMode is deliberately on. It double-invokes effects in development,
  // which is exactly what catches the polling and keyboard listeners in this app
  // failing to clean themselves up: a duplicated status poll or a second
  // keydown handler that decides two touches per keystroke.
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
