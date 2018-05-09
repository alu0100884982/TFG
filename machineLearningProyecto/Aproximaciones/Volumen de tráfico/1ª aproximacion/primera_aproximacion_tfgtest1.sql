CREATE TYPE tipo_fila_trafficvolume AS (tollgate int,direction int, left_side_interval timestamp );
CREATE TYPE par AS (tollgate char(1), direction int);

    
CREATE OR REPLACE FUNCTION create_firstrow_par_interval(parintervalo tipo_fila_trafficvolume) 
    RETURNS void AS $$
    DECLARE
    BEGIN

           UPDATE   tabla_resultado_traffic_volume AS thistable
                     SET twenty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_test1 othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '20 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval)
                    AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction AND
                    thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
                    
         
           UPDATE   tabla_resultado_traffic_volume AS thistable
                    SET forty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_test1 othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '40 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '20 minute') AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction  AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
          
                     
           UPDATE   tabla_resultado_traffic_volume AS thistable
                    SET sixty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_test1 othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '60 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '40 minute') AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction  AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND thistable.direction = parintervalo.direction;
          
          
           UPDATE   tabla_resultado_traffic_volume AS thistable
                    SET eighty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_test1 othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '80 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '60 minute')  AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction  AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
                   
                    
           UPDATE   tabla_resultado_traffic_volume AS thistable
                    SET onehundred_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_test1 othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '100 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '80 minute') AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
                    
           
           UPDATE   tabla_resultado_traffic_volume AS thistable
                    SET onehundredtwenty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_test1 othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '120 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '100 minute')  AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
    
    
    END;
    $$ LANGUAGE plpgsql;
    
CREATE OR REPLACE FUNCTION actualizar_filaactual_con_filaanterior_trafficvolume(rutaintervalo_actual tipo_fila_trafficvolume, rutaintervalo_anterior tipo_fila_trafficvolume) 
    RETURNS void AS $$
    DECLARE
        valor float;
     
    BEGIN
    
       SELECT P.* INTO valor
        FROM dblink('dbname=tfgdatosmodificados port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT round(AVG(volume)) FROM traffic_volume_tollgates_modified
        WHERE tollgate_id = '''|| rutaintervalo_anterior.tollgate|| ''' AND direction = '|| rutaintervalo_anterior.direction || ' AND
        time_window[1].time = '''|| (rutaintervalo_anterior.left_side_interval).time||''' AND
        time_window[2].time = ''' || (rutaintervalo_anterior.left_side_interval).time + INTERVAL '20 minute'|| ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (rutaintervalo_anterior.left_side_interval).date)) AS P( 
        volume float);
      
       UPDATE tabla_resultado_traffic_volume
                     SET twenty_min_previous = valor
                     WHERE time_window[1] = rutaintervalo_actual.left_side_interval AND  time_window[2] = (rutaintervalo_actual.left_side_interval + INTERVAL '20 minute')
                    AND  tollgate_id = rutaintervalo_actual.tollgate AND  direction = rutaintervalo_actual.direction;     
      UPDATE tabla_resultado_traffic_volume AS actual
                     SET forty_min_previous = before.twenty_min_previous,
                         sixty_min_previous = before.forty_min_previous,
                         eighty_min_previous = before.sixty_min_previous,
                         onehundred_min_previous = before.eighty_min_previous,
                         onehundredtwenty_min_previous = before.onehundred_min_previous
                     FROM tabla_resultado_traffic_volume before
                     WHERE actual.tollgate_id = rutaintervalo_actual.tollgate AND actual.direction = rutaintervalo_actual.direction AND actual.time_window[1]= rutaintervalo_actual.left_side_interval AND before.tollgate_id = rutaintervalo_anterior.tollgate AND before.direction = rutaintervalo_anterior.direction AND before.time_window[1] =  rutaintervalo_anterior.left_side_interval;
    END;
    $$ LANGUAGE plpgsql;
    
DO $$
<<block>>
DECLARE
  termina boolean DEFAULT FALSE;
  pares_intervalos tipo_fila_trafficvolume ARRAY;
  parintervalo_anterior tipo_fila_trafficvolume;
  contador integer DEFAULT 1;
  pares par ARRAY;
  tiempos time ARRAY;
  par par;
  tiempo time; 
BEGIN

pares_intervalos := ARRAY(SELECT '(' ||tollgate_id || ', ' || direction || ', ' || time_window[1] || ')' FROM  tabla_resultado_traffic_volume  WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')
ORDER BY tollgate_id, direction, time_window);
  
 WHILE contador <= ARRAY_LENGTH(pares_intervalos, 1) LOOP
       termina := FALSE;
       PERFORM create_firstrow_par_interval(pares_intervalos[contador]);
       parintervalo_anterior = pares_intervalos[contador];
       contador = contador + 1;
        WHILE NOT(termina) LOOP
            IF (pares_intervalos[contador].tollgate = parintervalo_anterior.tollgate AND pares_intervalos[contador].direction = parintervalo_anterior.direction AND  (pares_intervalos[contador].left_side_interval - parintervalo_anterior.left_side_interval) = INTERVAL '20 min') THEN 
                PERFORM actualizar_filaactual_con_filaanterior_trafficvolume(pares_intervalos[contador], parintervalo_anterior);
                parintervalo_anterior = pares_intervalos[contador];
                contador := contador + 1;
            ELSE 
                termina := TRUE;
            END IF;
        END LOOP;
   END LOOP;
   
CREATE OR REPLACE VIEW tiempo_con_intervalos_a_predecir_trafficvolume AS SELECT *
FROM weather_data_test1 JOIN tabla_resultado_traffic_volume
ON date_ = time_window[1].date AND CEIL(EXTRACT(HOUR FROM time_window[1])/3) * 3 = hour
ORDER BY tollgate_id, direction, time_window;

pares := ARRAY(SELECT DISTINCT(tollgate_id, direction) 
FROM tabla_resultado_traffic_volume
ORDER BY (tollgate_id, direction));

tiempos := ARRAY(SELECT DISTINCT(time_window[1].time)
FROM  tabla_resultado_traffic_volume
WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')
ORDER BY time_window[1].time);

FOREACH par IN ARRAY pares LOOP
     FOREACH tiempo IN ARRAY tiempos LOOP
        EXECUTE('CREATE TABLE  volume_' || par.tollgate ||'_' ||par.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || ' AS 
                SELECT EXTRACT(isodow FROM time_window[1].date) AS type_day, twenty_min_previous, forty_min_previous, sixty_min_previous, eighty_min_previous, onehundred_min_previous, onehundredtwenty_min_previous, pressure,sea_pressure,wind_direction,wind_speed,temperature,rel_humidity,precipitation,volume FROM tiempo_con_intervalos_a_predecir_trafficvolume  WHERE tollgate_id = ''' || par.tollgate|| ''' AND direction = '|| par.direction || 'AND time_window[1].time = ''' ||
                tiempo || ''' ORDER BY tollgate_id, direction, time_window');
                
        EXECUTE('UPDATE volume_' || par.tollgate ||'_' ||par.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || ' SET type_day = 1
                WHERE type_day BETWEEN 1 AND 5');
        EXECUTE('UPDATE volume_' || par.tollgate ||'_' ||par.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || ' SET type_day = 0
                WHERE type_day IN (6,7)');
     END LOOP;
END LOOP;

END block $$;

