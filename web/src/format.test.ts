import { describe, expect, it } from "vitest";
import {
  etaClockTime,
  formatDayDate,
  formatMinutes,
  formatMonth,
} from "./format";

describe("formatMinutes", () => {
  it("formats sub-hour trips", () => {
    expect(formatMinutes(0)).toBe("0 min");
    expect(formatMinutes(12)).toBe("12 min");
    expect(formatMinutes(59.4)).toBe("59 min");
    expect(formatMinutes(59.6)).toBe("1 h");
  });

  it("formats hour-plus trips", () => {
    expect(formatMinutes(60)).toBe("1 h");
    expect(formatMinutes(75)).toBe("1 h 15 min");
    expect(formatMinutes(180)).toBe("3 h");
  });

  it("handles invalid input", () => {
    expect(formatMinutes(NaN)).toBe("unknown");
    expect(formatMinutes(-5)).toBe("unknown");
  });
});

describe("etaClockTime", () => {
  it("adds minutes across hour boundaries", () => {
    expect(etaClockTime(8 * 60 + 30, 42)).toBe("09:12");
  });

  it("wraps past midnight", () => {
    expect(etaClockTime(23 * 60 + 50, 30)).toBe("00:20");
  });
});

describe("formatMonth", () => {
  it("formats YYYY-MM", () => {
    expect(formatMonth("2019-10")).toBe("Oct 2019");
    expect(formatMonth("2021-12")).toBe("Dec 2021");
  });
});

describe("formatDayDate", () => {
  it("formats ISO dates like Tue, Oct 1 2019", () => {
    expect(formatDayDate("2019-10-01")).toBe("Tue, Oct 1 2019");
    expect(formatDayDate("2020-04-07")).toBe("Tue, Apr 7 2020");
  });
});
