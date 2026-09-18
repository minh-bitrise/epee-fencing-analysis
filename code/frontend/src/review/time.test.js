import { describe, expect, it } from 'vitest'
import { formatClock, formatTime } from './time.js'

describe('video positions', () => {
  it('matches how the player writes the same moment', () => {
    // The player says 1:28 where the interface used to say 88.0s, so a user
    // matching a listed touch against the picture converted it in their head,
    // once per item, on a screen made for going through items one at a time.
    expect(formatTime(88.2)).toBe('1:28.2')
    expect(formatClock(88.2)).toBe('1:28')
  })

  it('pads the seconds, so 2:05 never reads as 2:5', () => {
    expect(formatTime(125.0)).toBe('2:05.0')
    expect(formatClock(125.0)).toBe('2:05')
  })

  it('keeps the tenth, which the labelling was careful about', () => {
    // A lunge lasts about ten frames and labels are placed to the frame.
    expect(formatTime(19.66)).toBe('0:19.6')
    expect(formatTime(0.4)).toBe('0:00.4')
  })

  it('does not go negative', () => {
    // Frame-stepping backwards past the start produces a small negative.
    expect(formatTime(-0.5)).toBe('0:00.0')
  })

  it('handles the minute boundary', () => {
    expect(formatTime(59.95)).toBe('0:59.9')
    expect(formatTime(60)).toBe('1:00.0')
  })
})
