import { getDB, initDB } from './db.js';
import { runFullSync } from './services/scheduler.js';
import { createSnapshot } from './createSnapshot.js';

// Runs from a clean checkout: initDB seeds from the committed snapshot, then the sync downloads each fund's
// full history. Clearing last_updated stops the sync skipping funds the snapshot marks as recently updated,
// which would otherwise leave them with no NAV history and blank metrics.
initDB();
getDB().prepare('UPDATE funds SET last_updated = NULL').run();
await runFullSync();
createSnapshot();
process.exit(0);
