CREATE TYPE tipo_fila AS (intersection char(1),tollgate int, left_side_interval timestamp );
CREATE OR REPLACE FUNCTION create_firstrow_route_interval(ruta-intervalo tipo_fila) 
    RETURNS void AS $$
    DECLARE
    BEGIN
           UPDATE tabla_resultado_average_travel_time 
                     SET twenty_min_previous = othertable.avg_travel_time
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (ruta-intervalo.left_side_interval - '20 minute') AND othertable.time_window[2] = (ruta-intervalo.left_side_interval)
                    AND othertable.intersection_id = ruta_intervalo.intersection AND othertable.tollgate_id = ruta_intervalo.tollgate;
           UPDATE tabla_resultado_average_travel_time 
                    SET forty_min_previous = AVG(othertable.avg_travel_time)
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (ruta-intervalo.left_side_interval - '40 minute') AND othertable.time_window[2] = (ruta-intervalo.left_side_interval - '20 minute')
                    AND othertable.intersection_id = ruta_intervalo.intersection AND othertable.tollgate_id = ruta_intervalo.tollgate;
           UPDATE tabla_resultado_average_travel_time 
                    SET sixty_min_previous = AVG(othertable.avg_travel_time)
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (ruta-intervalo.left_side_interval - '60 minute') AND othertable.time_window[2] = (ruta-intervalo.left_side_interval - '40 minute')
                    AND othertable.intersection_id = ruta_intervalo.intersection AND othertable.tollgate_id = ruta_intervalo.tollgate;
           UPDATE tabla_resultado_average_travel_time 
                    SET eighty_min_previous = AVG(othertable.avg_travel_time)
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (ruta-intervalo.left_side_interval - '80 minute') AND othertable.time_window[2] = (ruta-intervalo.left_side_interval)
                    AND othertable.intersection_id = ruta_intervalo.intersection AND othertable.tollgate_id = ruta_intervalo.tollgate;
           UPDATE tabla_resultado_average_travel_time 
                    SET onehundred_min_previous = AVG(othertable.avg_travel_time)
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (ruta-intervalo.left_side_interval - '100 minute') AND othertable.time_window[2] = (ruta-intervalo.left_side_interval)
                    AND othertable.intersection_id = ruta_intervalo.intersection AND othertable.tollgate_id = ruta_intervalo.tollgate;
           UPDATE tabla_resultado_average_travel_time 
                    SET onehundredtwenty_min_previous = AVG(othertable.avg_travel_time)
                    FROM travel_time_intersection_to_tollgate_test1 othertable
                    WHERE othertable.time_window[1] = (ruta-intervalo.left_side_interval - '120 minute') AND othertable.time_window[2] = (ruta-intervalo.left_side_interval)
                    AND othertable.intersection_id = ruta_intervalo.intersection AND othertable.tollgate_id = ruta_intervalo.tollgate;
 = 
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
  ruta_intervalo_anterior tipo_fila;
  contador integer DEFAULT 1;
BEGIN 
  rutas_intervalos := ARRAY(SELECT '(' ||intersection_id || ', ' || tollgate_id || ', ' || time_window[1] || ')' FROM tabla_resultado_average_travel_time ORDER BY intersection_id, tollgate_id, time_window);
  
  WHILE contador <= ARRAY_LENGTH(rutas_intervalos, 1) LOOP
       termina := FALSE;
       PERFORM create_firstrow_route_interval(rutas_intervalos[contador]);
       ruta_intervalo_anterior = rutas_intervalos[contador];
       contador = contador + 1;
        WHILE NOT(termina) LOOP
            IF ((rutas_intervalos[contador] - ruta_intervalo_anterior) = INTERVAL '20 min') THEN
                actualizar_filaactual_con_filaanterior(rutas_intervalos[contador], ruta_intervalo_anterior)
                contador := contador + 1;
            ELSE 
                termina := TRUE
            END IF;
        END LOOP;
  END LOOP;
      
END block $$;

/**
   EXECUTE('CREATE TABLE ' || ruta-intervalo || '(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN (\'A\', \'B\', \'C\')),
        tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
        time_window timestamp ARRAY[2],
        twenty_min_previous float,
        forty_min_previous float,
        sixty_min_previous float,
        eighty_min_previous float,
        onehundred_min_previous float,
        onehundredandtwenty_min_previous float,
        avg_travel_time float)';**/






