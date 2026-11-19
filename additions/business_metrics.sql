-- New descriptive marts. Source fact grain = store × product × day.
-- All rates must be reaggregated as numerator / denominator, never averaged rates.
-- sale_amount is globally rescaled, NOT RMB, GMV or margin.
CREATE OR REPLACE TEMP VIEW enriched AS
SELECT d.*, s.hours_stock_status[22] AS close_oos
FROM int_daily d JOIN stg_sales s USING (store_id,product_id,dt);

CREATE OR REPLACE TEMP VIEW mart_store_day AS
SELECT dt,city_id,store_id,COUNT(*) AS product_days,
       SUM(oos_hours) AS oos_hours,
       SUM((oos_hours>2)::INT) AS severe_days,
       SUM(close_oos) AS close_oos_days,
       SUM(sale_amount) AS scaled_sales
FROM enriched GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW mart_category_day AS
SELECT dt,city_id,first_category_id,COUNT(*) AS product_days,
       SUM(oos_hours) AS oos_hours,SUM((oos_hours>2)::INT) AS severe_days
FROM enriched GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW mart_hour_city AS
SELECT h.dt,s.city_id,h.hour,COUNT(*) AS observations,
       SUM(h.stock_status) AS oos_observations
FROM int_hourly h JOIN stg_sales s USING(store_id,product_id,dt)
WHERE h.in_window GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW mart_persistence AS
SELECT CASE WHEN t.label_dt>='2024-06-26' THEN 'test' ELSE 'development' END AS label_period,
       (d.oos_hours>2) AS current_severe,COUNT(*) AS observations,
       SUM(t.label) AS next_severe,
       AVG(t.label) AS next_severe_rate
FROM int_daily d JOIN int_target t USING(store_id,product_id,dt)
GROUP BY ALL;
