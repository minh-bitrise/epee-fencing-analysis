import { render, screen } from '@testing-library/react'
import { useEffect, useRef, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import Tabs from './Tabs.jsx'

describe('Tabs', () => {
  it('shows the first tab, and switches on click', async () => {
    const { container } = render(
      <Tabs tabs={{ One: () => <p>first</p>, Two: () => <p>second</p> }} />)
    expect(screen.getByText('first')).toBeInTheDocument()
    container.querySelectorAll('button')[1].click()
    expect(await screen.findByText('second')).toBeInTheDocument()
  })

  it('does not remount the active tab when the parent re-renders', async () => {
    // The defect this exists for. The caller builds its `tabs` object inline, so
    // those arrow functions are new objects on every parent render. Mounting
    // them as <Body /> made the tab body a new component TYPE each time and
    // React remounted the lot. The parent re-renders about four times a second
    // while the video plays, and each remount re-ran the panels' effects: one
    // bout left playing issued roughly 150 requests for its profile.
    const mounted = vi.fn()

    function Panel() {
      useEffect(() => { mounted() }, [])
      return <p>panel</p>
    }

    function Parent() {
      const [n, setN] = useState(0)
      const ref = useRef(null)
      ref.current = () => setN((x) => x + 1)
      return (
        <>
          <button onClick={() => ref.current()}>tick</button>
          <span>{n}</span>
          {/* deliberately inline, exactly as ReviewView does it */}
          <Tabs tabs={{ Bout: () => <Panel />, Notes: () => <p>notes</p> }} />
        </>
      )
    }

    render(<Parent />)
    expect(mounted).toHaveBeenCalledTimes(1)
    const tick = screen.getByText('tick')
    tick.click(); tick.click(); tick.click()
    expect(await screen.findByText('3')).toBeInTheDocument()
    expect(mounted).toHaveBeenCalledTimes(1)
  })
})
