
CREATE TYPE time_pair AS (first_time time, second_time time );
CREATE TYPE route AS (
    intersection_id  text,
    tollgate_id      integer
);

CREATE OR REPLACE VIEW weather_byday AS SELECT  date_, AVG(pressure) as pressure, AVG(sea_pressure) as sea_pressure, AVG(wind_direction) as wind_direction, AVG(wind_speed) as wind_speed, AVG(temperature) as temperature, AVG(rel_humidity) as rel_humidity, AVG(precipitation) as precipitation
FROM weather_data_modified GROUP by date_
ORDER BY date_;


CREATE OR REPLACE FUNCTION create_view(ruta route, intervalo time_pair) 
    RETURNS void AS $$
    DECLARE
    BEGIN
        EXECUTE('CREATE OR REPLACE VIEW ' || ruta.intersection_id || '_' || ruta.tollgate_id || '_' ||TO_CHAR(intervalo.first_time, 'HH24_MI')||'_'|| TO_CHAR(intervalo.second_time, 'HH24_MI')|| ' AS SELECT *
FROM weather_byday w JOIN travel_time_intersection_to_tollgate_modified t
ON w.date_ = t.time_window[1].date 
WHERE intersection_id = ''' || ruta.intersection_id || ''' AND tollgate_id = ' || ruta.tollgate_id || ' AND time_window[1].time = ''' || intervalo.first_time || ''' AND time_window[2].time =  ''' || 
intervalo.second_time || ''';');

    END;
    $$ LANGUAGE plpgsql;

DO $$
<<block>>
DECLARE
   rutas  route[6];
   ruta route;
   time_aux text ARRAY;
   time_aux2 text ARRAY;
   time_string text;
   time_1 time_pair;
   intervals time_pair ARRAY DEFAULT '{}';
    
    
BEGIN 

       rutas[1] = ('A',2);
       rutas[2] = ('A',3);
       rutas[3] = ('B',1);
       rutas[4] = ('B',3);
       rutas[5] = ('C',1);
       rutas[6] = ('C',3);  
       time_aux:= ARRAY(SELECT DISTINCT(time_window[1].time, time_window[2].time)
                FROM travel_time_intersection_to_tollgate_modified); 
       
       FOREACH time_string IN ARRAY time_aux LOOP
          time_string := regexp_replace(time_string, '[()]', '');
          time_aux2 := STRING_TO_ARRAY(time_string, ',');
          time_1.first_time := time_aux2[1];
          time_1.second_time := time_aux2[2];
          intervals := ARRAY_APPEND(intervals, time_1);
       END LOOP;
       
       FOREACH ruta IN ARRAY rutas LOOP
          FOREACH time_1 IN ARRAY intervals LOOP
                PERFORM create_view(ruta,time_1);
          END LOOP;
       END LOOP;
       
END block $$;

