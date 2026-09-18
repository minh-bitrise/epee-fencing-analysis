import { useCallback, useState } from 'react'
import ErrorBoundary from './ErrorBoundary.jsx'
import UploadView from './upload/UploadView.jsx'
import ReviewView from './review/ReviewView.jsx'

/**
 * The two halves of the workflow: get a bout processed, then review it.
 *
 * They are separate views rather than one page because they are separated in
 * time. Processing a three minute clip takes minutes, and the whole point of
 * the job layer is that the user is not made to sit and watch it. Review starts
 * when the work is finished, possibly in a later session.
 *
 * Which view is showing is held here rather than in a router. Two views, one
 * transition between them, and no URLs anyone needs to share or bookmark: a
 * router would be a dependency and a layer of indirection bought with nothing.
 */
export default function App() {
  const [view, setView] = useState('upload')
  const [openBoutId, setOpenBoutId] = useState(null)

  // Handing a bout id across is what closes the loop the design describes. Until
  // this existed a user could process a video and then had to find it again in a
  // dropdown by an id derived from its output directory.
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
