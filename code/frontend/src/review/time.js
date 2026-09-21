/* Positions in the video, written the way the video writes them: 1:28.2 for a
   timestamp a user reads against a player, 1:28 where the tenth is noise. */

const mmss = (s) => {
  const t = Math.max(0, s)
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`
}

/** A position, to a tenth: 1:28.2 */
export const formatTime = (s) =>
  `${mmss(s)}.${Math.floor((Math.max(0, s) % 1) * 10)}`

/** A position, to the second: 1:28. For axis labels and ranges. */
export const formatClock = mmss
