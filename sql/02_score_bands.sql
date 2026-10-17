-- Risk deciles on the held-out TEST set. Decile 1 = lowest scores = riskiest customers.
WITH t AS (
    SELECT s.score, s.pd, s.defaulted,
           NTILE(10) OVER (ORDER BY s.score ASC) AS risk_decile
    FROM scored s WHERE s.split = 'test'
)
SELECT risk_decile,
       COUNT(*)                          AS customers,
       ROUND(MIN(score), 0)              AS min_score,
       ROUND(MAX(score), 0)              AS max_score,
       SUM(defaulted)                      AS defaults,
       ROUND(100.0 * AVG(defaulted), 1)    AS actual_default_pct,
       ROUND(100.0 * AVG(pd), 1)         AS predicted_pd_pct
FROM t GROUP BY risk_decile ORDER BY risk_decile;
