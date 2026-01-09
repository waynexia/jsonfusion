-- Basic JSONFUSION insert and query
CREATE TABLE users (id INT, data JSONFUSION);
INSERT INTO users VALUES
  (1, '{"name":"Ada","age":39,"active":true,"meta":{"city":"NY"}}'),
  (2, '{"name":"Bob","age":40,"active":false,"meta":{"city":"SF"}}'),
  (3, '{"name":"Eve"}');
SELECT id, data FROM users ORDER BY id;
SELECT data.name AS name, data.meta.city AS city FROM users ORDER BY id;
