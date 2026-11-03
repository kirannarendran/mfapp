import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';

const BENCHMARK_CODES = [100484, 120137];
const MAX_NAV_AGE_DAYS = 10; // Long enough to cover market holidays
const MIN_COVERAGE_RATIO = 0.99;

function readStats(path) {
  const db = new Database(path, { readonly: true });
  try {
    return {
      integrity: db.pragma('integrity_check', { simple: true }),
      funds: db.prepare('SELECT COUNT(*) AS n FROM funds').get().n,
      withCagr3y: db.prepare('SELECT COUNT(*) AS n FROM fund_metrics WHERE cagr_3y IS NOT NULL').get().n,
      withSharpe: db.prepare('SELECT COUNT(*) AS n FROM fund_metrics WHERE sharpe IS NOT NULL').get().n,
      latestNav: db.prepare('SELECT MAX(last_nav_date) AS d FROM funds').get().d,
      users: db.prepare('SELECT COUNT(*) AS n FROM users').get().n,
      benchmarks: BENCHMARK_CODES.map(code => ({
        code,
        ...db.prepare('SELECT COUNT(*) AS rows, MAX(date) AS latest FROM nav_history WHERE scheme_code = ?').get(code),
      })),
    };
  } finally {
    db.close();
  }
}

/**
 * Compare a new snapshot with the previous one. Returns the reasons it must not be published.
 */
export function validateSnapshot(newPath, previousPath, today = new Date()) {
  const next = readStats(newPath);
  const prev = readStats(previousPath);
  const failures = [];

  if (next.integrity !== 'ok') failures.push(`integrity check failed: ${next.integrity}`);
  if (next.users !== 0) failures.push(`contains ${next.users} user records`);
  for (const key of ['funds', 'withCagr3y', 'withSharpe']) {
    if (next[key] < prev[key] * MIN_COVERAGE_RATIO) failures.push(`${key} dropped from ${prev[key]} to ${next[key]}`);
  }
  if (next.latestNav < prev.latestNav) failures.push(`latest NAV went backwards: ${prev.latestNav} to ${next.latestNav}`);

  const oldestAllowed = new Date(today.getTime() - MAX_NAV_AGE_DAYS * 24 * 3600 * 1000).toISOString().split('T')[0];
  for (const bench of next.benchmarks) {
    const before = prev.benchmarks.find(b => b.code === bench.code);
    if (bench.rows < before.rows) failures.push(`benchmark ${bench.code} lost rows: ${before.rows} to ${bench.rows}`);
    if (!bench.latest || bench.latest < oldestAllowed) failures.push(`benchmark ${bench.code} is stale: ${bench.latest}`);
  }
  return failures;
}

// Run directly if invoked from CLI: node server/validateSnapshot.js <new.db> <previous.db>
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const failures = validateSnapshot(process.argv[2], process.argv[3]);
  if (failures.length > 0) {
    console.error(`[Validate] ❌ Not publishing:\n - ${failures.join('\n - ')}`);
    process.exit(1);
  }
  console.log('[Validate] ✅ Snapshot passed all checks');
}
