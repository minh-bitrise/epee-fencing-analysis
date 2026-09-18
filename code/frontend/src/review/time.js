/**
 * Positions in the video, written the way the video writes them.
 *
 * WHY. The player's own readout says 1:28, and every time the interface
 * produced elsewhere said 88.0s, so a user matching a listed touch or lunge
 * against the picture had to convert in their head, once per item, on a screen
 * whose whole purpose is going through items one at a time.
 *
 * The tenth is kept because it is load-bearing: a lunge lasts about ten frames
 * and labels are placed to the frame, so rounding to the second would throw away
 * the precision the labelling was careful about. Durations are a different
 * quantity and keep their own formatter in ReviewQueue.
 */

const mmss = (s) => {
  const t = Math.max(0, s)
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`
}

/** A position, to a tenth: 1:28.2 */
export const formatTime = (s) =>
  `${mmss(s)}.${Math.floor((Math.max(0, s) % 1) * 10)}`

/** A position, to the second: 1:28. For axis labels and ranges. */
export const formatClock = mmss
