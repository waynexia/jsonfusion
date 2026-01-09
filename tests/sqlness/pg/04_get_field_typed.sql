-- Explicit get_field_typed usage
CREATE TABLE explicit_demo (id INT, data JSONFUSION);
INSERT INTO explicit_demo VALUES (1, '{"name":"Ada","age":39}');
SELECT
  get_field_typed(data, 'name') AS name_json,
  get_field_typed(data, 'age', CAST(NULL AS INT)) AS age_typed
FROM explicit_demo;
