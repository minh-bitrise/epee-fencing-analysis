import { Component } from 'react'

/* Stops one broken panel taking the whole application down. WHY THIS EXISTS. */
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
