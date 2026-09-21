/* A short label that opens to the reasoning behind a number. WHY THE PROSE IS FOLDED AWAY. */
export default function Disclosure({ label = 'Why?', children }) {
  return (
    <details className="disclose">
      <summary>{label}</summary>
      <div>{children}</div>
    </details>
  )
}
