import { useCallback, useState } from 'react'
import ErrorBoundary from './ErrorBoundary.jsx'
import UploadView from './upload/UploadView.jsx'
import ReviewView from './review/ReviewView.jsx'

/* The two halves of the workflow: get a bout processed, then review it. */
export default function App() {
  const [view, setView] = useState('upload')
  const [openBoutId, setOpenBoutId] = useState(null)

  // Handing a bout id across is what closes the loop the design describes.
  const openInReview = useCallback((boutId) => {
    setOpenBoutId(boutId)
    setView('review')
  }, [])

  return (
    <>
      <header>
        <h1>Epee Bout Analysis</h1>
        <nav>
          <button className={view === 'upload' ? 'active' : ''}
                  onClick={() => setView('upload')}>Upload and process</button>
          <button className={view === 'review' ? 'active' : ''}
                  onClick={() => setView('review')}>Review</button>
        </nav>
      </header>
      {view === 'upload'
        ? <ErrorBoundary label="The upload view">
            <UploadView onOpenBout={openInReview} />
          </ErrorBoundary>
        : <ErrorBoundary label="The review view">
            <ReviewView initialBoutId={openBoutId} />
          </ErrorBoundary>}
    </>
  )
}
