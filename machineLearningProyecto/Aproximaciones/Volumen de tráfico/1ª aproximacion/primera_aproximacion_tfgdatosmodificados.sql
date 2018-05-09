CREATE TYPE tipo_fila_trafficvolume AS (tollgate int,direction int, left_side_interval timestamp );
CREATE TYPE par AS (tollgate char(1), direction int);

CREATE OR REPLACE FUNCTION checkAttributeValue(valores text[], parintervalo tipo_fila_trafficvolume) 
RETURNS void AS $$
    DECLARE
        valor float;
        tiempo1 time;
        tiempo2 time;
    BEGIN
        tiempo1 = (parintervalo.left_side_interval).time - CAST(valores[2] AS int) * INTERVAL '1 minute';
        tiempo2 = (parintervalo.left_side_interval).time - CAST(valores[2] AS int) * INTERVAL '1 minute' + INTERVAL '20 minute';
        
        EXECUTE('SELECT ' || valores[1] || ' FROM  traffic_volume_tollgates_modified  WHERE time_window[1] = ''' ||(parintervalo.left_side_interval)||''' AND time_window[2] = ''' ||         (parintervalo.left_side_interval + INTERVAL '20 minute') || ''' AND tollgate_id = ''' || parintervalo.tollgate || ''' AND direction = ' || parintervalo.direction) INTO valor; 
        IF (valor IS NULL) THEN
         EXECUTE('SELECT round(AVG(volume)) FROM  traffic_volume_tollgates_modified
        WHERE tollgate_id = '''|| parintervalo.tollgate|| ''' AND direction = '|| parintervalo.direction || ' AND
        time_window[1].time = '''|| tiempo1|| ''' AND
        time_window[2].time = ''' || tiempo2 || ''' AND EXTRACT(isodow FROM time_window[1].date) = '     ||      EXTRACT(isodow FROM (parintervalo.left_side_interval).date)) INTO valor;
         EXECUTE('UPDATE   traffic_volume_tollgates_modified SET ' || valores[1] || ' = ' || valor || 
                     ' WHERE time_window[1] = '''|| parintervalo.left_side_interval || ''' AND  time_window[2] = ''' || (parintervalo.left_side_interval  + INTERVAL '20 minute')
                     || ''' AND tollgate_id = ''' ||parintervalo.tollgate|| ''' AND  direction = ' || parintervalo.direction);
        
        
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    
CREATE OR REPLACE FUNCTION create_firstrow_par_interval(parintervalo tipo_fila_trafficvolume) 
    RETURNS void AS $$
    DECLARE
    BEGIN

           UPDATE   traffic_volume_tollgates_modified AS thistable
                     SET twenty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_modified othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '20 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval)
                    AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction AND
                    thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
                    
                    PERFORM checkAttributeValue(array['twenty_min_previous','20']::text[], parintervalo::tipo_fila_trafficvolume );
         
           UPDATE   traffic_volume_tollgates_modified AS thistable
                    SET forty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_modified othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '40 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '20 minute') AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction  AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
          
                     PERFORM checkAttributeValue(array['forty_min_previous','40']::text[], parintervalo);
                     
           UPDATE   traffic_volume_tollgates_modified AS thistable
                    SET sixty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_modified othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '60 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '40 minute') AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.tollgate  AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND thistable.direction = parintervalo.direction;
                    
                     PERFORM checkAttributeValue(array['sixty_min_previous','60']::text[], parintervalo);
          
          
           UPDATE   traffic_volume_tollgates_modified AS thistable
                    SET eighty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_modified othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '80 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '60 minute')  AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.tollgate  AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
                   
                     PERFORM checkAttributeValue(array['eighty_min_previous','80']::text[], parintervalo);
                    
           UPDATE   traffic_volume_tollgates_modified AS thistable
                    SET onehundred_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_modified othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '100 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '80 minute') AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
                    
           
                    PERFORM checkAttributeValue(array['onehundred_min_previous','100']::text[], parintervalo);
           
           UPDATE   traffic_volume_tollgates_modified AS thistable
                    SET onehundredtwenty_min_previous = othertable.volume
                    FROM  traffic_volume_tollgates_modified othertable
                    WHERE othertable.time_window[1] = (parintervalo.left_side_interval - INTERVAL '120 minute') AND othertable.time_window[2] = (parintervalo.left_side_interval - INTERVAL '100 minute')  AND othertable.tollgate_id = parintervalo.tollgate AND othertable.direction = parintervalo.direction AND
                     thistable.time_window[1] = parintervalo.left_side_interval AND  thistable.time_window[2] = (parintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.tollgate_id = parintervalo.tollgate AND  thistable.direction = parintervalo.direction;
    
                    PERFORM checkAttributeValue(array['onehundredtwenty_min_previous','120']::text[], parintervalo);
    
    END;
    $$ LANGUAGE plpgsql;
    
CREATE OR REPLACE FUNCTION actualizar_filaactual_con_filaanterior_trafficvolume(parintervalo_actual tipo_fila_trafficvolume, parintervalo_anterior tipo_fila_trafficvolume) 
    RETURNS void AS $$
    DECLARE
        valor float;
    BEGIN
    
       SELECT volume INTO valor FROM  traffic_volume_tollgates_modified 
       WHERE time_window[1] = (parintervalo_anterior.left_side_interval) AND time_window[2] = (parintervalo_anterior.left_side_interval  + INTERVAL '20 minute')
       AND tollgate_id = parintervalo_anterior.tollgate AND direction = parintervalo_anterior.direction;
        
       IF (valor IS NULL) THEN
        EXECUTE('SELECT round(AVG(volume)) INTO valor FROM  traffic_volume_tollgates_modified
        WHERE tollgate_id = '''|| parintervalo_anterior.tollgate|| ''' AND direction = '|| parintervalo_anterior.direction || ' AND
        time_window[1].time = '''|| (parintervalo_anterior.left_side_interval).time||''' AND
        time_window[2].time = ''' || (parintervalo_anterior.left_side_interval).time + INTERVAL '20 minute'|| ''' AND EXTRACT(isodow FROM time_window[1].date) = ' || EXTRACT(isodow FROM (parintervalo_anterior.left_side_interval).date));
       END IF;
        
       UPDATE   traffic_volume_tollgates_modified AS thistable
                     SET twenty_min_previous = valor
                     WHERE time_window[1] = parintervalo_actual.left_side_interval AND  time_window[2] = (parintervalo_actual.left_side_interval  + INTERVAL '20 minute')
                     AND tollgate_id = parintervalo_actual.tollgate AND  direction = parintervalo_actual.direction;
                        
      
      UPDATE   traffic_volume_tollgates_modified AS actual
                     SET twenty_min_previous = before.volume,
                         forty_min_previous = before.twenty_min_previous,
                         sixty_min_previous = before.forty_min_previous,
                         eighty_min_previous = before.sixty_min_previous,
                         onehundred_min_previous = before.eighty_min_previous,
                         onehundredtwenty_min_previous = before.onehundred_min_previous
                     FROM   traffic_volume_tollgates_modified before
                     WHERE actual.tollgate_id = parintervalo_actual.tollgate AND actual.direction = parintervalo_actual.direction AND actual.time_window[1]=                      parintervalo_actual.left_side_interval AND before.tollgate_id = parintervalo_anterior.tollgate AND before.direction = parintervalo_anterior.direction AND before.time_window[1] =      parintervalo_anterior.left_side_interval;
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

pares_intervalos := ARRAY(SELECT '(' ||tollgate_id || ', ' || direction || ', ' || time_window[1] || ')' FROM  traffic_volume_tollgates_modified  WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')
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
   
CREATE OR REPLACE VIEW tiempo_con_intervalos_trafficvolume AS SELECT *
FROM weather_data_modified JOIN (SELECT *
FROM  traffic_volume_tollgates_modified 
WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00')  OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')
ORDER BY tollgate_id, direction, time_window
) t ON date_ = time_window[1].date AND CEIL(EXTRACT(HOUR FROM time_window[1])/3) * 3 = hour
ORDER BY tollgate_id, direction, time_window;

pares := ARRAY(SELECT DISTINCT(tollgate_id, direction)
FROM  traffic_volume_tollgates_modified
WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')
ORDER BY (tollgate_id, direction));

tiempos := ARRAY(SELECT DISTINCT(time_window[1].time)
FROM  traffic_volume_tollgates_modified
WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00')
ORDER BY time_window[1].time);

FOREACH par IN ARRAY pares LOOP
     FOREACH tiempo IN ARRAY tiempos LOOP
        EXECUTE('CREATE TABLE  volume_' || par.tollgate ||'_' ||par.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || ' AS 
                SELECT EXTRACT(isodow FROM time_window[1].date) AS type_day, twenty_min_previous, forty_min_previous, sixty_min_previous, eighty_min_previous, onehundred_min_previous, onehundredtwenty_min_previous, pressure,sea_pressure,wind_direction,wind_speed,temperature,rel_humidity,precipitation,volume FROM tiempo_con_intervalos_trafficvolume  WHERE tollgate_id = ''' || par.tollgate|| ''' AND direction = '|| par.direction || 'AND time_window[1].time = ''' ||
                tiempo || ''' ORDER BY tollgate_id, direction, time_window');
                
        EXECUTE('UPDATE volume_' || par.tollgate ||'_' ||par.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || ' SET type_day = 1
                WHERE type_day BETWEEN 1 AND 5');
        EXECUTE('UPDATE volume_' || par.tollgate ||'_' ||par.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo) || ' SET type_day = 0
                WHERE type_day IN (6,7)');
     END LOOP;
END LOOP;

END block $$;

