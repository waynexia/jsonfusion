-- JSONFUSION type hints and usage
CREATE TABLE typed (id INT, data JSONFUSION (age INT NOT NULL, active BOOLEAN));
INSERT INTO typed VALUES (1, '{"age":21,"active":true}');
INSERT INTO typed VALUES (2, '{"age":"nope","active":"yes"}');
SELECT data.age + 1 AS age_plus_one,
       CASE WHEN data.active THEN 1 ELSE 0 END AS active_flag
FROM typed;
