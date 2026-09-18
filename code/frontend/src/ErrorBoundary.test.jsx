import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ErrorBoundary from './ErrorBoundary.jsx'

function Boom() {
  throw new TypeError("Cannot read properties of undefined (reading '1')")
}

describe('ErrorBoundary', () => {
  it('passes children through when nothing throws', () => {
    render(<ErrorBoundary label="Panel"><p>fine</p></ErrorBoundary>)
    expect(screen.getByText('fine')).toBeInTheDocument()
  })

  it('names the panel and keeps the message instead of blanking', () => {
    // The real failure: a reprocess job recorded its piste region as a boolean,
    // one card indexed it as an array, and the whole interface went blank. A
    // blank page destroys the evidence, so the requirement is that the thing
    // that broke is named and the error survives on screen.
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(<ErrorBoundary label="The upload view"><Boom /></ErrorBoundary>)
    expect(screen.getByText(/The upload view could not be displayed/)).toBeInTheDocument()
    expect(screen.getByText(/reading '1'/)).toBeInTheDocument()
    spy.mockRestore()
  })
})
