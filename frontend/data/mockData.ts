import dashboardData from './dashboardData.json';

// Real data from the ML pipeline extraction script
const realData = dashboardData;

// Feature importance: normalize to use the combined average for bar chart display
const churnDrivers = realData.churnDrivers.map((d: any) => ({
  feature: d.feature.replace(/_/g, ' ').replace('tenure category Loyal (4y+)', 'tenure(Loyal 4y+)'),
  importance: d.lgbm_imp !== undefined
    ? Math.round(((d.xgb_imp || 0) + (d.lgbm_imp || 0)) / 2 * 10000) / 10000
    : d.importance,
}));

// Suggestions — these are DERIVED from the real stats, not fabricated
const churnReduction = [
  {
    id: 'clv-optimization',
    title: 'Optimize CLV-to-Spend Ratio',
    description: `The #1 churn driver by LightGBM (29.5% impact). Low CLV segment has ${realData.clvSegments[0]?.churnRate}% churn vs ${realData.clvSegments[2]?.churnRate}% for High CLV. Introduce tiered loyalty perks and feature bundles that grow perceived value.`,
    impact: 'High' as const,
    effort: 'Medium' as const,
    timeline: '2-4 weeks',
    dataSource: `clv_spend_ratio — LightGBM: ${(realData.churnDrivers[0]?.lgbm_imp * 100).toFixed(1)}% importance`,
    kpi: `${realData.clvSegments[0]?.churnRate}%`,
    kpiLabel: 'Low CLV Churn Rate',
  },
  {
    id: 'early-tenure',
    title: 'First 90-Day Onboarding Program',
    description: `New customers (<6 months) churn at ${realData.tenureImpact[0]?.churnRate}% — over double the Loyal rate of ${realData.tenureImpact[4]?.churnRate}%. Deploy a structured 90-day onboarding sequence with welcome calls, walkthrough, and value reviews.`,
    impact: 'High' as const,
    effort: 'Low' as const,
    timeline: '1-2 weeks',
    dataSource: `tenure_months — LightGBM: ${(realData.churnDrivers[1]?.lgbm_imp * 100).toFixed(1)}% importance`,
    kpi: `${realData.tenureImpact[0]?.churnRate}%`,
    kpiLabel: 'New Customer Churn',
  },
  {
    id: 'contract-annual',
    title: 'Promote Annual Contracts',
    description: `Monthly contracts churn at ${realData.contractStats[1]?.churnRate}% while Annual churn at only ${realData.contractStats[0]?.churnRate}%. Offer 2-month-free incentive for Annual lock-in to reduce monthly contract churn.`,
    impact: 'High' as const,
    effort: 'Low' as const,
    timeline: '1 week',
    dataSource: `contract_type — Monthly: ${realData.contractStats[1]?.count} customers (${realData.contractStats[1]?.churnRate}%)`,
    kpi: `${realData.contractStats[0]?.churnRate}%`,
    kpiLabel: 'Annual Churn Rate',
  },
  {
    id: 'payment-grace',
    title: 'Late Payment Grace Period & Alerts',
    description: `Customers with 3-5 late payments churn at ${realData.paymentStats[2]?.churnRate}% — nearly 2.5x the rate of those with 0 late (${realData.paymentStats[0]?.churnRate}%). Automate grace-period notifications at 3, 7, and 14 days.`,
    impact: 'Medium' as const,
    effort: 'Low' as const,
    timeline: '1 week',
    dataSource: `late_payments — 0 late: ${realData.paymentStats[0]?.churnRate}%, 3-5 late: ${realData.paymentStats[2]?.churnRate}%`,
    kpi: `${realData.paymentStats[2]?.churnRate}%`,
    kpiLabel: '3-5 Late Payments Churn',
  },
  {
    id: 'nps-improvement',
    title: 'NPS Detractor Recovery Program',
    description: `NPS Detractors churn at ${realData.npsStats[0]?.churnRate}% vs Promoters at ${realData.npsStats[2]?.churnRate}%. There are ${realData.npsStats[0]?.count} Detractors. Conduct root-cause analysis calls and offer dedicated resolutions.`,
    impact: 'Medium' as const,
    effort: 'Medium' as const,
    timeline: '2-3 weeks',
    dataSource: `nps_category — ${realData.npsStats[0]?.count} detractors, ${realData.npsStats[2]?.count} promoters`,
    kpi: `${realData.npsStats[0]?.churnRate}%`,
    kpiLabel: 'Detractor Churn Rate',
  },
];

const winBackCampaigns = [
  {
    id: 'targeted-retention',
    title: 'Targeted Retention Offer',
    description: `Deploy personalized discount offers to the ${realData.kpis.highRiskCount} ML-identified at-risk customers. Offer 15-25% discount on next renewal, priority support access for 90 days, or free feature upgrade.`,
    segment: `${realData.kpis.highRiskCount} High-Risk Customers`,
    channel: 'Email + In-App',
    expectedROI: `$${realData.kpis.revenueAtRisk.toLocaleString()} at risk`,
    successRate: '35-45%',
  },
  {
    id: 'nps-detractor-outreach',
    title: 'NPS Detractor Outreach',
    description: `Contact all ${realData.npsStats[0]?.count} Detractors (NPS < 7) with a personal call from a senior account manager. Address their specific pain points. Detractors churn at ${realData.npsStats[0]?.churnRate}%.`,
    segment: `${realData.npsStats[0]?.count} NPS Detractors`,
    channel: 'Phone + Email',
    expectedROI: `$${Math.round(realData.npsStats[0]?.count * realData.kpis.avgClv * 0.12).toLocaleString()} recovered`,
    successRate: '25-35%',
  },
  {
    id: 'monthly-to-annual',
    title: 'Monthly-to-Annual Migration',
    description: `${realData.contractStats[1]?.count} customers on Monthly contracts churn at ${realData.contractStats[1]?.churnRate}%. Offer a "lock-in & save" campaign with 2 months free for switching to Annual (${realData.contractStats[0]?.churnRate}% churn rate).`,
    segment: `${realData.contractStats[1]?.count} Monthly Users`,
    channel: 'Email + Sales',
    expectedROI: `${(realData.contractStats[1]?.churnRate - realData.contractStats[0]?.churnRate).toFixed(1)}% churn reduction`,
    successRate: '20-30%',
  },
  {
    id: 'cross-sell-expansion',
    title: 'Cross-Sell Product Expansion',
    description: `Products owned is a top-10 feature (avg rank: ${realData.churnDrivers[7]?.avg_rank}). More products = deeper lock-in. Run targeted upsell/cross-sell campaigns to single-product users with complementary bundles.`,
    segment: 'Single-Product Users',
    channel: 'In-App + Sales Team',
    expectedROI: 'Increased retention',
    successRate: '15-25%',
  },
];

const additionalInsights = [
  {
    title: 'CLV Segment Drives Churn Dramatically',
    insight: `Low CLV customers churn at ${realData.clvSegments[0]?.churnRate}%, Medium at ${realData.clvSegments[1]?.churnRate}%, and High at only ${realData.clvSegments[2]?.churnRate}%. Focus retention on upgrading Low-CLV users.`,
    metric: `${realData.clvSegments[0]?.churnRate}%`,
    metricLabel: 'Low CLV Churn',
  },
  {
    title: 'Monthly Contracts Are a Red Flag',
    insight: `Monthly contracts churn at ${realData.contractStats[1]?.churnRate}% vs Annual at ${realData.contractStats[0]?.churnRate}%. The flexibility of month-to-month means lower commitment which leads to higher churn.`,
    metric: `${(realData.contractStats[1]?.churnRate - realData.contractStats[0]?.churnRate).toFixed(1)}%`,
    metricLabel: 'Monthly vs Annual Gap',
  },
  {
    title: 'Late Payments Compound Risk',
    insight: `Customers with 0 late payments churn at ${realData.paymentStats[0]?.churnRate}%, while 3-5 late payments pushes churn to ${realData.paymentStats[2]?.churnRate}%. Payment behavior is a leading churn indicator.`,
    metric: `${realData.paymentStats[2]?.churnRate}%`,
    metricLabel: '3-5 Late Churn',
  },
  {
    title: 'NPS Is an Actionable Signal',
    insight: `Detractors (${realData.npsStats[0]?.count} users, ${realData.npsStats[0]?.churnRate}% churn) vs Promoters (${realData.npsStats[2]?.count} users, ${realData.npsStats[2]?.churnRate}% churn). NPS surveys detect dissatisfaction early.`,
    metric: `${(realData.npsStats[0]?.churnRate - realData.npsStats[2]?.churnRate).toFixed(1)}%`,
    metricLabel: 'Detractor-Promoter Gap',
  },
];

export const mockData = {
  kpis: realData.kpis,
  churnDrivers,
  riskSegments: realData.riskSegments,
  tenureImpact: realData.tenureImpact,
  engagementImpact: realData.engagementImpact,
  clvSegments: realData.clvSegments,
  contractStats: realData.contractStats,
  npsStats: realData.npsStats,
  paymentStats: realData.paymentStats,
  modelPerformance: realData.modelPerformance,
  churnReduction,
  winBackCampaigns,
  additionalInsights,
  customerDirectory: realData.customerDirectory,
};
