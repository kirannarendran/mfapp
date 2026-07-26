/**
 * Calculates the future value of a SIP investment (investment at beginning of month).
 * formula: FV = P * [ ((1 + i)^n - 1) / i ] * (1 + i)
 */
export const calculateSIPFutureValue = (monthlyInvestment, years, expectedReturnRate) => {
    if (!monthlyInvestment || !years || !expectedReturnRate) return { investedAmount: 0, estimatedReturns: 0, totalValue: 0 };

    const i = expectedReturnRate / 12 / 100;
    const n = years * 12;

    const investedAmount = monthlyInvestment * n;
    
    // Future value of annuity due (beginning of month)
    const fv = monthlyInvestment * ((Math.pow(1 + i, n) - 1) / i) * (1 + i);
    const totalValue = Math.round(fv);
    const estimatedReturns = totalValue - investedAmount;

    return {
        investedAmount,
        estimatedReturns,
        totalValue
    };
};

/**
 * Returns a recommended asset allocation and fund categories based on risk profile and time horizon.
 */
export const getAssetAllocation = (riskProfile, years) => {
    // Basic logic:
    // Low Risk: Debt heavy, Conservative Hybrid
    // Medium Risk: Flexi Cap, Large Cap, Balanced Advantage
    // High Risk: Mid Cap, Small Cap, Multi Cap

    if (years < 3) {
        return {
            allocation: "100% Debt / Liquid",
            categories: ["Liquid Fund", "Overnight Fund", "Low Duration"]
        };
    }

    switch (riskProfile) {
        case "High":
            return {
                allocation: "80% Equity (Mid/Small Cap), 20% Debt",
                categories: ["Mid Cap", "Small Cap", "Multi Cap", "Flexi Cap"]
            };
        case "Medium":
            return {
                allocation: "60% Equity (Large/Flexi), 40% Debt",
                categories: ["Large & Mid Cap", "Flexi Cap", "Large Cap", "Aggressive Hybrid"]
            };
        case "Low":
        default:
            return {
                allocation: "30% Equity, 70% Debt",
                categories: ["Conservative Hybrid", "Corporate Bond", "Large Cap", "Index Fund"]
            };
    }
};

/**
 * Filters the master fund list to find top matches for the recommended categories.
 * Prioritizes "Growth" plans and excludes "IDCW" (Dividend) to ensure better long-term compounding suggestions.
 */
export const getRecommendedFunds = (allFunds, categories) => {
    if (!allFunds || allFunds.length === 0 || !categories) return [];

    const recommendations = {};

    categories.forEach(category => {
        // Keyword matching
        const matches = allFunds.filter(fund => {
            const name = fund.schemeName.toLowerCase();
            const cat = category.toLowerCase();

            // Must match category name
            const isCategoryMatch = name.includes(cat);

            // Must be a Growth plan (exclude IDCW/Dividend)
            const isGrowth = name.includes("growth") && !name.includes("idcw") && !name.includes("dividend");

            // Preferred Fund Houses (Optional: to avoid obscure funds if many matches)
            // Just sorting by shorter name length often gives the main fund vs specific series
            return isCategoryMatch && isGrowth;
        });

        // Sort by name length (heuristic: "Axis Bluechip Fund - Growth" is likely the main one vs "Axis Bluechip Fund - Direct Plan - Growth Option")
        // And pick top 3 distinct fund houses if possible
        recommendations[category] = matches.slice(0, 5);
    });

    return recommendations;
};
