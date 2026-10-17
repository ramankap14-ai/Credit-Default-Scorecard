-- Default rate by segment (all 30,000 customers). Descriptive only: not used to fit the model.
SELECT 'latest_repayment_status' AS segment_type,
       CASE WHEN PAY_1 <= -1 THEN 'a. paid in full / no use'
            WHEN PAY_1 = 0 THEN 'b. revolving'
            WHEN PAY_1 = 1 THEN 'c. 1 month late'
            WHEN PAY_1 = 2 THEN 'd. 2 months late'
            ELSE 'e. 3+ months late' END AS segment,
       COUNT(*) AS customers, SUM(defaulted) AS defaults,
       ROUND(100.0 * AVG(defaulted), 1) AS default_rate_pct
FROM customers GROUP BY 2
UNION ALL
SELECT 'months_late_of_last_6',
       CASE WHEN n_late >= 3 THEN '3+' ELSE CAST(n_late AS TEXT) END,
       COUNT(*), SUM(defaulted), ROUND(100.0 * AVG(defaulted), 1)
FROM customers GROUP BY 2
UNION ALL
SELECT 'latest_utilisation',
       CASE WHEN util_1 = 0 THEN 'a. zero'
            WHEN util_1 < 0.3 THEN 'b. under 30%'
            WHEN util_1 < 0.6 THEN 'c. 30-60%'
            WHEN util_1 < 0.9 THEN 'd. 60-90%'
            ELSE 'e. 90%+' END,
       COUNT(*), SUM(defaulted), ROUND(100.0 * AVG(defaulted), 1)
FROM customers GROUP BY 2
UNION ALL
SELECT 'credit_limit',
       CASE WHEN LIMIT_BAL < 50000 THEN 'a. under 50k'
            WHEN LIMIT_BAL < 100000 THEN 'b. 50-100k'
            WHEN LIMIT_BAL < 200000 THEN 'c. 100-200k'
            ELSE 'd. 200k+' END,
       COUNT(*), SUM(defaulted), ROUND(100.0 * AVG(defaulted), 1)
FROM customers GROUP BY 2
ORDER BY 1, 2;
