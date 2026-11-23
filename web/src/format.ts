/** Pure formatting helpers (unit tested). */

/** 12 -> "12 min"; 75 -> "1 h 15 min". */
export function formatMinutes(minutes: number): string {
  if (!Number.isFinite(minutes) || minutes < 0) return "unknown";
  const m = Math.round(minutes);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem === 0 ? `${h} h` : `${h} h ${rem} min`;
}

/** ETA clock time: "08:30" + 42 min -> "09:12". */
export function etaClockTime(pickupMinuteOfDay: number, minutes: number): string {
  const total = Math.round(pickupMinuteOfDay + minutes);
  const wrapped = ((total % 1440) + 1440) % 1440;
  const h = Math.floor(wrapped / 60);
  const m = wrapped % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** "2019-10-01" -> "Tue, Oct 1 2019" (UTC weekday, no local drift). */
export function formatDayDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  const wd = WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${wd}, ${months[m - 1]} ${d} ${y}`;
}

/** "2019-10" -> "Oct 2019". */
export function formatMonth(ym: string): string {
  const [y, m] = ym.split("-").map(Number);
  const names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${names[m - 1]} ${y}`;
}
