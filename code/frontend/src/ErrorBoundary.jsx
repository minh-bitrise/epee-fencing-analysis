import { Component } from 'react'

/**
 * Stops one broken panel taking the whole application down.
 *
 * WHY THIS EXISTS. A reprocess job records its piste region as a boolean flag
 * rather than the polygon, one card indexed it as an array, and the entire
 * interface went blank: the page rendered and then disappeared. Everything else
 * on the screen was working. There was simply nothing between a thrown render
 * and the root.
 *
 * A blank page is the worst possible failure here, because it destroys the
 * evidence. The user cannot see which panel broke, cannot reach the others, and
 * the only symptom is that the application appears to have died. Anything that
 * keeps the rest of the screen alive and names the part that failed is better,
 * however plain it looks.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Kept on the console deliberately: this is the only place the stack
    // survives, and the visible message is short by design.
    console.error(`panel "${this.props.label || 'unknown'}" failed`, error, info)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="note err">
        <b>{this.props.label || 'This panel'} could not be displayed.</b>{' '}
        {String(this.state.error.message || this.state.error)}
        {' '}The rest of the page is unaffected; the details are in the browser
        console.
      </div>
    )
  }
}
