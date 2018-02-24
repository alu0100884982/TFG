CREATE TYPE tipo_fila AS (intersection char(1),tollgate int, left_side_interval timestamp );
CREATE OR REPLACE FUNCTION create_firstrow_route_interval(rutaintervalo tipo_fila) 
    RETURNS void AS $$
    DECLARE
    BEGIN
           UPDATE tabla_resultado_average_travel_time AS thistable
                     SET twenty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo.left_side_interval - INTERVAL '20 minute') AND othertable.time_window[2] = (rutaintervalo.left_side_interval)
                    AND othertable.intersection_id = rutaintervalo.intersection AND othertable.tollgate_id = rutaintervalo.tollgate AND
                    thistable.time_window[1] = rutaintervalo.left_side_interval AND  thistable.time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.intersection_id = rutaintervalo.intersection AND  thistable.tollgate_id = rutaintervalo.tollgate;
         
           UPDATE tabla_resultado_average_travel_time AS thistable
                    SET forty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo.left_side_interval - INTERVAL '40 minute') AND othertable.time_window[2] = (rutaintervalo.left_side_interval - INTERVAL '20 minute') AND othertable.intersection_id = rutaintervalo.intersection AND othertable.tollgate_id = rutaintervalo.tollgate  AND
                     thistable.time_window[1] = rutaintervalo.left_side_interval AND  thistable.time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute')
                    AND thistable.intersection_id = rutaintervalo.intersection AND  thistable.tollgate_id = rutaintervalo.tollgate;
          
           UPDATE tabla_resultado_average_travel_time AS thistable
                    SET sixty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo.left_side_interval - INTERVAL '60 minute') AND othertable.time_window[2] = (rutaintervalo.left_side_interval - INTERVAL '40 minute') AND othertable.intersection_id = rutaintervalo.intersection AND othertable.tollgate_id = rutaintervalo.tollgate  AND
                     thistable.time_window[1] = rutaintervalo.left_side_interval AND  thistable.time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.intersection_id = rutaintervalo.intersection AND thistable.tollgate_id = rutaintervalo.tollgate;
                    
           UPDATE tabla_resultado_average_travel_time AS thistable
                    SET eighty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo.left_side_interval - INTERVAL '80 minute') AND othertable.time_window[2] = (rutaintervalo.left_side_interval - INTERVAL '60 minute')  AND othertable.intersection_id = rutaintervalo.intersection AND othertable.tollgate_id = rutaintervalo.tollgate  AND
                     thistable.time_window[1] = rutaintervalo.left_side_interval AND  thistable.time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.intersection_id = rutaintervalo.intersection AND  thistable.tollgate_id = rutaintervalo.tollgate;
                    
           UPDATE tabla_resultado_average_travel_time AS thistable
                    SET onehundred_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo.left_side_interval - INTERVAL '100 minute') AND othertable.time_window[2] = (rutaintervalo.left_side_interval - INTERVAL '80 minute') AND othertable.intersection_id = rutaintervalo.intersection AND othertable.tollgate_id = rutaintervalo.tollgate AND
                     thistable.time_window[1] = rutaintervalo.left_side_interval AND  thistable.time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.intersection_id = rutaintervalo.intersection AND  thistable.tollgate_id = rutaintervalo.tollgate;
                    
           UPDATE tabla_resultado_average_travel_time AS thistable
                    SET onehundredtwenty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo.left_side_interval - INTERVAL '120 minute') AND othertable.time_window[2] = (rutaintervalo.left_side_interval - INTERVAL '100 minute')  AND othertable.intersection_id = rutaintervalo.intersection AND othertable.tollgate_id = rutaintervalo.tollgate AND
                     thistable.time_window[1] = rutaintervalo.left_side_interval AND  thistable.time_window[2] = (rutaintervalo.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.intersection_id = rutaintervalo.intersection AND  thistable.tollgate_id = rutaintervalo.tollgate;
    END;
    $$ LANGUAGE plpgsql;
    
CREATE OR REPLACE FUNCTION actualizar_filaactual_con_filaanterior(rutaintervalo_actual tipo_fila, rutaintervalo_anterior tipo_fila) 
    RETURNS void AS $$
    DECLARE
    BEGIN
   
    
    UPDATE tabla_resultado_average_travel_time AS thistable
                     SET twenty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (rutaintervalo_actual.left_side_interval - INTERVAL '20 minute') AND othertable.time_window[2] = (rutaintervalo_actual.left_side_interval)
                    AND othertable.intersection_id = rutaintervalo_actual.intersection AND othertable.tollgate_id = rutaintervalo_actual.tollgate AND
                     thistable.time_window[1] = rutaintervalo_actual.left_side_interval AND  thistable.time_window[2] = (rutaintervalo_actual.left_side_interval + INTERVAL '20 minute')
                    AND  thistable.intersection_id = rutaintervalo_actual.intersection AND  thistable.tollgate_id = rutaintervalo_actual.tollgate;
                    
      UPDATE tabla_resultado_average_travel_time AS actual
                     SET forty_min_previous = before.twenty_min_previous,
                         sixty_min_previous = before.forty_min_previous,
                         eighty_min_previous = before.sixty_min_previous,
                         onehundred_min_previous = before.eighty_min_previous,
                         onehundredtwenty_min_previous = before.onehundred_min_previous
                     FROM tabla_resultado_average_travel_time before
                     WHERE actual.intersection_id = rutaintervalo_actual.intersection AND actual.tollgate_id = rutaintervalo_actual.tollgate AND actual.time_window[1]=                      rutaintervalo_actual.left_side_interval AND before.intersection_id = rutaintervalo_anterior.intersection AND before.tollgate_id = rutaintervalo_anterior.tollgate AND before.time_window[1] =      rutaintervalo_anterior.left_side_interval;
    END;
    $$ LANGUAGE plpgsql;
/**
CREATE OR REPLACE FUNCTION create_columns_intervals()    
RETURNS void AS $$
    DECLARE
    BEGIN
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column twenty_min_previous float;
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column forty_min_previous float;
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column sixty_min_previous float;
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column eighty_min_previous float;
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column onehundred_min_previous float;
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column onehundredtwenty_min_previous float;
        ALTER TABLE tabla_resultado_average_travel_time 
        DROP column avg_travel_time;
        ALTER TABLE tabla_resultado_average_travel_time 
        ADD column avg_travel_time;
    END;
    $$ LANGUAGE plpgsql;
**/
DO $$
<<block>>
DECLARE
  termina boolean DEFAULT FALSE;
  rutas_intervalos tipo_fila ARRAY;
  rutaintervalo_anterior tipo_fila;
  contador integer DEFAULT 1;
BEGIN 
  rutas_intervalos := ARRAY(SELECT '(' ||intersection_id || ', ' || tollgate_id || ', ' || time_window[1] || ')' FROM tabla_resultado_average_travel_time ORDER BY intersection_id, tollgate_id, time_window);
  
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
      
END block $$;

/**
   EXECUTE('CREATE TABLE ' || rutaintervalo || '(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN (\'A\', \'B\', \'C\')),
        tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
        time_window timestamp ARRAY[2],
        twenty_min_previous float,
        forty_min_previous float,
        sixty_min_previous float,
        eighty_min_previous float,
        onehundred_min_previous float,
        onehundredandtwenty_min_previous float,
        avg_travel_time float)';**/






