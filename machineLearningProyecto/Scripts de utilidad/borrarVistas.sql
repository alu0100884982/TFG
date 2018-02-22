

SELECT 'DROP VIEW ' || table_name || ';'
FROM information_schema.views
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
AND table_name !~ '^pg_';

DO $$
<<block>>
DECLARE
   vistas text[];
   vista text;
   
BEGIN
   vistas := ARRAY(SELECT table_name || ';'
                FROM information_schema.views
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_name !~ '^pg_');
                
   FOREACH vista IN ARRAY vistas LOOP
         IF (vista != 'weather_byday;') THEN
           EXECUTE('DROP VIEW ' || vista);
         END IF;
   END LOOP;
END block $$;
