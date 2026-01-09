-- Schema merging across inserts
CREATE TABLE merge_demo (id INT, data JSONFUSION);
INSERT INTO merge_demo VALUES (1, '{"a":1}');
INSERT INTO merge_demo VALUES (2, '{"b":"two"}');
SELECT id, json_display(data) AS data FROM merge_demo ORDER BY id;
SELECT data.a AS a, data.b AS b FROM merge_demo ORDER BY id;
