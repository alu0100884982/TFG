CREATE TYPE tipo_fila AS (intersection char(1),tollgate int, left_side_interval timestamp );
CREATE TYPE ruta AS (intersection char(1), tollgate int);



CREATE OR REPLACE FUNCTION create_firstrow_route_interval(rutaintervalo tipo_fila) 
    RETURNS void AS $$
    DECLARE
    valor float;
    BEGIN
           SELECT P.* INTO valor
           FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT AVG(avg_travel_time)      FROM travel_time_intersection_to_tollgate_modified
        WHERE intersection_id = '''|| rutaintervalo.intersection|| ''' AND tollgate_id = '|| rutaintervalo.tollgate || ' AND
        time_window[1].time = '''|| (rutaintervalo.left_side_interval - INTERVAL '20 minute').time||''' AND
        time_window[2].time = ''' || (rutaintervalo.left_side_interval).time || ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo.left_side_interval).date)) AS P( 
        avg_travel_time float);
        
         UPDATE  travel_time_intersection_to_tollgate_test1 
                    SET twenty_min_previous = valor
                    WHERE time_window[1] = rutaintervalo.left_side_interval AND  time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute') AND intersection_id = rutaintervalo.intersection AND tollgate_id = rutaintervalo.tollgate;
          
           SELECT P.* INTO valor
           FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT AVG(avg_travel_time)      FROM travel_time_intersection_to_tollgate_modified
        WHERE intersection_id = '''|| rutaintervalo.intersection|| ''' AND tollgate_id = '|| rutaintervalo.tollgate || ' AND
        time_window[1].time = '''|| (rutaintervalo.left_side_interval - INTERVAL '40 minute').time||''' AND
        time_window[2].time = ''' || (rutaintervalo.left_side_interval - INTERVAL '20 minute').time || ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo.left_side_interval).date)) AS P( 
        avg_travel_time float);
      
        UPDATE  travel_time_intersection_to_tollgate_test1 
                    SET forty_min_previous = valor
                    WHERE time_window[1] = rutaintervalo.left_side_interval AND  time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute') AND intersection_id = rutaintervalo.intersection AND  tollgate_id = rutaintervalo.tollgate;
      
         SELECT P.* INTO valor
           FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT AVG(avg_travel_time)      FROM travel_time_intersection_to_tollgate_modified
        WHERE intersection_id = '''|| rutaintervalo.intersection|| ''' AND tollgate_id = '|| rutaintervalo.tollgate || ' AND
        time_window[1].time = '''|| (rutaintervalo.left_side_interval - INTERVAL '60 minute').time||''' AND
        time_window[2].time = ''' || (rutaintervalo.left_side_interval - INTERVAL '40 minute').time || ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo.left_side_interval).date)) AS P( 
        avg_travel_time float);
        
          UPDATE  travel_time_intersection_to_tollgate_test1 
                    SET sixty_min_previous = valor
                    WHERE time_window[1] = rutaintervalo.left_side_interval AND  time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute') AND intersection_id = rutaintervalo.intersection AND tollgate_id = rutaintervalo.tollgate;
      
          SELECT P.* INTO valor
           FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT AVG(avg_travel_time)      FROM travel_time_intersection_to_tollgate_modified
        WHERE intersection_id = '''|| rutaintervalo.intersection|| ''' AND tollgate_id = '|| rutaintervalo.tollgate || ' AND
        time_window[1].time = '''|| (rutaintervalo.left_side_interval - INTERVAL '80 minute').time||''' AND
        time_window[2].time = ''' || (rutaintervalo.left_side_interval - INTERVAL '60 minute').time || ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo.left_side_interval).date)) AS P( 
        avg_travel_time float);
        
         UPDATE  travel_time_intersection_to_tollgate_test1 
                    SET eighty_min_previous = valor
                    WHERE time_window[1] = rutaintervalo.left_side_interval AND  time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute') AND intersection_id = rutaintervalo.intersection AND tollgate_id = rutaintervalo.tollgate;
      
          SELECT P.* INTO valor
           FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT AVG(avg_travel_time)      FROM travel_time_intersection_to_tollgate_modified
        WHERE intersection_id = '''|| rutaintervalo.intersection|| ''' AND tollgate_id = '|| rutaintervalo.tollgate || ' AND
        time_window[1].time = '''|| (rutaintervalo.left_side_interval - INTERVAL '100 minute').time||''' AND
        time_window[2].time = ''' || (rutaintervalo.left_side_interval - INTERVAL '80 minute').time || ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo.left_side_interval).date)) AS P( 
        avg_travel_time float);
        
         UPDATE  travel_time_intersection_to_tollgate_test1 
                    SET onehundred_min_previous = valor
                    WHERE time_window[1] = rutaintervalo.left_side_interval AND  time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute') AND intersection_id = rutaintervalo.intersection AND  tollgate_id = rutaintervalo.tollgate;
      
          SELECT P.* INTO valor
           FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT AVG(avg_travel_time)      FROM travel_time_intersection_to_tollgate_modified
        WHERE intersection_id = '''|| rutaintervalo.intersection|| ''' AND tollgate_id = '|| rutaintervalo.tollgate || ' AND
        time_window[1].time = '''|| (rutaintervalo.left_side_interval - INTERVAL '120 minute').time||''' AND
        time_window[2].time = ''' || (rutaintervalo.left_side_interval - INTERVAL '100 minute').time || ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo.left_side_interval).date)) AS P( 
        avg_travel_time float);
        
         UPDATE  travel_time_intersection_to_tollgate_test1 
                    SET onehundredtwenty_min_previous = valor
                    WHERE time_window[1] = rutaintervalo.left_side_interval AND  time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute') AND intersection_id = rutaintervalo.intersection AND tollgate_id = rutaintervalo.tollgate;
      
         
    
    END;
    $$ LANGUAGE plpgsql;
    
CREATE OR REPLACE FUNCTION actualizar_filaactual_con_filaanterior(rutaintervalo_actual tipo_fila, rutaintervalo_anterior tipo_fila) 
    RETURNS void AS $$
    DECLARE
        valor float;
    BEGIN
    
       SELECT avg_travel_time INTO valor FROM travel_time_intersection_to_tollgate_test1 
       WHERE time_window[1] = (rutaintervalo_anterior.left_side_interval) AND time_window[2] = (rutaintervalo_anterior.left_side_interval  + INTERVAL '20 minute')
       AND intersection_id = rutaintervalo_anterior.intersection AND tollgate_id = rutaintervalo_anterior.tollgate;
        
       UPDATE  travel_time_intersection_to_tollgate_test1 AS thistable
                     SET twenty_min_previous = valor
                     WHERE time_window[1] = rutaintervalo_actual.left_side_interval AND  time_window[2] = (rutaintervalo_actual.left_side_interval  + INTERVAL '20 minute')
                     AND intersection_id = rutaintervalo_actual.intersection AND  tollgate_id = rutaintervalo_actual.tollgate;
                        
      
      UPDATE  travel_time_intersection_to_tollgate_test1 AS actual
                     SET forty_min_previous = before.twenty_min_previous,
                         sixty_min_previous = before.forty_min_previous,
                         eighty_min_previous = before.sixty_min_previous,
                         onehundred_min_previous = before.eighty_min_previous,
                         onehundredtwenty_min_previous = before.onehundred_min_previous
                     FROM  travel_time_intersection_to_tollgate_test1 before
                     WHERE actual.intersection_id = rutaintervalo_actual.intersection AND actual.tollgate_id = rutaintervalo_actual.tollgate AND actual.time_window[1]=                      rutaintervalo_actual.left_side_interval AND before.intersection_id = rutaintervalo_anterior.intersection AND before.tollgate_id = rutaintervalo_anterior.tollgate AND before.time_window[1] =      rutaintervalo_anterior.left_side_interval;
    END;
    $$ LANGUAGE plpgsql;
    
DO $$
<<block>>
DECLARE
   termina boolean DEFAULT FALSE;
  rutas_intervalos tipo_fila ARRAY;
  rutaintervalo_anterior tipo_fila;
  contador integer DEFAULT 1;
  routes ruta ARRAY;
  tiempos time ARRAY;
  route ruta;
  tiempo time; 
BEGIN
        
rutas_intervalos := ARRAY(SELECT '(' ||intersection_id||  ', ' || tollgate_id || ', ' || time_window[1] || ')' FROM  travel_time_intersection_to_tollgate_test1
ORDER BY intersection_id, tollgate_id, time_window);
 WHILE contador <= ARRAY_LENGTH(rutas_intervalos, 1) LOOP
       termina := FALSE;
       PERFORM create_firstrow_route_interval(rutas_intervalos[contador]);
       rutaintervalo_anterior = rutas_intervalos[contador];
       contador = contador + 1;
        WHILE NOT(termina) LOOP
            IF (rutas_intervalos[contador].intersection = rutaintervalo_anterior.intersection AND rutas_intervalos[contador].tollgate = rutaintervalo_anterior.tollgate AND  (rutas_intervalos[contador].left_side_interval - rutaintervalo_anterior.left_side_interval) = INTERVAL '20 min') THEN 
                PERFORM actualizar_filaactual_con_filaanterior(rutas_intervalos[contador], rutaintervalo_anterior);
                 rutaintervalo_anterior = rutas_intervalos[contador];
                contador := contador + 1;
            ELSE 
                termina := TRUE;
            END IF;
        END LOOP;
   END LOOP;
   
CREATE OR REPLACE VIEW tiempo_con_intervalos AS SELECT *
FROM weather_data_test1 JOIN (SELECT *
FROM travel_time_intersection_to_tollgate_test1 
ORDER BY intersection_id, tollgate_id, time_window
) t ON date_ = time_window[1].date AND CEIL(EXTRACT(HOUR FROM time_window[1])/3) * 3 = hour
ORDER BY intersection_id, tollgate_id, time_window;

routes := ARRAY(SELECT DISTINCT(intersection_id, tollgate_id)
FROM travel_time_intersection_to_tollgate_test1
ORDER BY (intersection_id, tollgate_id));

tiempos := ARRAY(SELECT DISTINCT(time_window[1].time)
FROM travel_time_intersection_to_tollgate_test1
ORDER BY time_window[1].time);

FOREACH route IN ARRAY routes LOOP
     FOREACH tiempo IN ARRAY tiempos LOOP
        EXECUTE('CREATE TABLE  ' || route.intersection ||'_' ||route.tollgate || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || '_twohoursprevious AS 
                SELECT EXTRACT(isodow FROM time_window[1].date) AS type_day, twenty_min_previous, forty_min_previous, sixty_min_previous, eighty_min_previous, onehundred_min_previous, onehundredtwenty_min_previous, pressure,sea_pressure,wind_direction,wind_speed,temperature,rel_humidity,precipitation,avg_travel_time FROM tiempo_con_intervalos  WHERE intersection_id = ''' || route.intersection|| ''' AND tollgate_id = '|| route.tollgate || 'AND time_window[1].time = ''' ||
                tiempo || ''' ORDER BY intersection_id, tollgate_id, time_window');
                
        EXECUTE('UPDATE ' || route.intersection ||'_' ||route.tollgate || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || '_twohoursprevious SET type_day = 1
                WHERE type_day BETWEEN 1 AND 5');
        EXECUTE('UPDATE ' || route.intersection ||'_' ||route.tollgate || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || '_twohoursprevious SET type_day = 0
                WHERE type_day IN (6,7)');
     END LOOP;
END LOOP;

END block $$;



/** Before executing the script it is necessary to create the following columns:
DO $$
<<block>>
    DECLARE
    BEGIN
        ALTER TABLE travel_time_intersection_to_tollgate_test1 
        ADD column twenty_min_previous float;
        ALTER TABLE travel_time_intersection_to_tollgate_test1 
        ADD column forty_min_previous float;
        ALTER TABLE travel_time_intersection_to_tollgate_test1
        ADD column sixty_min_previous float;
        ALTER TABLE travel_time_intersection_to_tollgate_test1
        ADD column eighty_min_previous float;
        ALTER TABLE travel_time_intersection_to_tollgate_test1
        ADD column onehundred_min_previous float;
        ALTER TABLE travel_time_intersection_to_tollgate_test1 
        ADD column onehundredtwenty_min_previous float;
    END $$;


 **/


