/**
 * A short label that opens to the reasoning behind a number.
 *
 * WHY THE PROSE IS FOLDED AWAY. This project's explanations are worth keeping:
 * nearly every caveat in the interface was paid for by a measurement that went
 * wrong, and removing them would mean showing figures whose limits are invisible.
 * But printing all of them at once produced a screen that was mostly paragraphs,
 * and a caveat in a wall of text is not read any more carefully than one that is
 * absent.
 *
 * Folded, they are one click from the number they qualify and out of the way of
 * the work. The trigger is always present, so nothing is hidden, only quiet.
 */
export default function Disclosure({ label = 'Why?', children }) {
  return (
    <details className="disclose">
      <summary>{label}</summary>
      <div>{children}</div>
    </details>
  )
}
