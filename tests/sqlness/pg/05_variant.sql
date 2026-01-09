-- Variant support with mixed types
CREATE TABLE variant_demo (id INT, data JSONFUSION);
INSERT INTO variant_demo VALUES
  (1, '{"value":1,"meta":{"tag":"a"}}'),
  (2, '{"value":"two","meta":{"tag":2}}'),
  (3, '{"value":{"nested":3},"meta":{"tag":true}}'),
  (4, '{"value":[1,2,3],"meta":{"tag":["x","y"]}}');
SELECT id, data FROM variant_demo ORDER BY id;
SELECT id, data.value AS value FROM variant_demo ORDER BY id;
SELECT id, data.meta.tag AS tag FROM variant_demo ORDER BY id;
