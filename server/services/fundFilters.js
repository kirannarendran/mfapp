// SQL predicate (funds aliased as `f`) for schemes people can actually invest in. Hides side pockets,
// which have a zero NAV or a "(Segregated ..." name, and "Unclaimed" plans holding uninvested payouts.
// Main schemes named "(Existing Number of Segregated Portfolios - 1)" are real funds and stay visible.
// Use it for discovery (search, screener, AI picks), not for looking up what a user already holds.
export const INVESTABLE_FUND_SQL = `(
  COALESCE(f.last_nav, 1) > 0
  AND f.scheme_name NOT LIKE '%Unclaimed%'
  AND f.scheme_name NOT LIKE '%(Segregated%'
)`;
