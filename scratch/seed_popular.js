import { syncNavData } from '../server/services/dataSync.js';
import { computeAndStoreMetrics } from '../server/services/metricsEngine.js';
import { initDB } from '../server/db.js';

const popularFunds = [
  // Large Cap
  120465, // HDFC Top 100
  118274, // ICICI Prudential Bluechip
  119062, // SBI Bluechip
  112292, // Nippon India Large Cap
  
  // Mid Cap
  118989, // HDFC Mid-Cap Opportunities
  118269, // ICICI Prudential Midcap
  120716, // Kotak Emerging Equity
  
  // Small Cap
  125497, // SBI Small Cap
  125354, // Nippon India Small Cap
  
  // Flexi Cap
  122639, // Parag Parikh Flexi Cap (already have)
  122171, // HDFC Flexi Cap
  120824, // Kotak Flexi Cap
  
  // ELSS
  120846, // Kotak Tax Saver
  119239, // SBI Long Term Equity
];

async function seed() {
  console.log("Seeding popular funds...");
  initDB();
  for (const code of popularFunds) {
    try {
      console.log(`Syncing NAV for ${code}...`);
      await syncNavData(code);
      console.log(`Computing metrics for ${code}...`);
      await computeAndStoreMetrics(code);
    } catch (e) {
      console.error(`Failed for ${code}: ${e.message}`);
    }
  }
  console.log("Seeding complete.");
}

seed();
