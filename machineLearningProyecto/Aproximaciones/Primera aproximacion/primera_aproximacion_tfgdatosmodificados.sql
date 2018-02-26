
CREATE TYPE time_pair AS (first_time time, second_time time );
CREATE TYPE route AS (
    intersection_id  text,
    tollgate_id      integer
);

/*
CREATE OR REPLACE FUNCTION create_view(ruta route, intervalo time_pair) 
    RETURNS void AS $$
    DECLARE
    BEGIN
    END;
    $$ LANGUAGE plpgsql;
*/
DO $$
<<block>>
DECLARE
    
BEGIN 
SELECT *
FROM travel_time_intersection_to_tollgate_modified 
WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')S
ORDER BY intersection_id, tollgate_id, time_window;
 
END block $$;

