-- Profit if we approve only the safest N deciles (test set).
-- Unit economics (ASSUMPTION): an approved good account earns 1 unit; an approved bad account loses R units.
WITH t AS (
    SELECT s.defaulted, NTILE(10) OVER (ORDER BY s.score DESC) AS safety_decile   -- 1 = safest
    FROM scored s WHERE s.split = 'test'
),
agg AS (
    SELECT safety_decile, COUNT(*) AS n, SUM(defaulted) AS bads FROM t GROUP BY safety_decile
),
cum AS (
    SELECT safety_decile,
           SUM(n)    OVER (ORDER BY safety_decile) AS approved,
           SUM(bads) OVER (ORDER BY safety_decile) AS bads
    FROM agg
)
SELECT safety_decile AS deciles_approved,
       approved,
       bads AS defaults,
       ROUND(100.0 * bads / approved, 1)               AS default_rate_pct,
       (approved - bads) - 2 * bads                    AS profit_R2,
       (approved - bads) - 3 * bads                    AS profit_R3,
       (approved - bads) - 4 * bads                    AS profit_R4,
       (approved - bads) - 6 * bads                    AS profit_R6
FROM cum ORDER BY safety_decile;
