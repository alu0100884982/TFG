CREATE TABLE tabla_resultado_traffic_volume (tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window varchar(50),
direction smallint CONSTRAINT has_direction_value CHECK (direction IN (0,1)),
volume int);

CREATE OR REPLACE FUNCTION insert_rows(fecha_inicial timestamp, fecha_final timestamp) 
    RETURNS void AS $$
    DECLARE
            tollgates integer[3] DEFAULT '{1,2,3}';
            directions integer[2] DEFAULT '{0,1}';
            tollgate_id integer;
            direction integer;
            aux timestamp;
            contador integer;
    BEGIN
      FOREACH tollgate_id IN ARRAY tollgates 
      LOOP
        FOREACH direction IN ARRAY directions
        LOOP
          aux := fecha_inicial;
          WHILE aux != fecha_final LOOP
             FOR contador in 1..7 LOOP
                 INSERT INTO  tabla_resultado_traffic_volume VALUES(tollgate_id, ARRAY[aux, aux + '20 minute'],direction,NULL);
                  aux :=  aux + '1 day';
             END LOOP;
             aux := aux + '-7 days';
             aux := aux + '20 minute';
          END LOOP;
       END LOOP;
      END LOOP;
    END;
    $$ LANGUAGE plpgsql;


DO $$
<<another_block2>>
DECLARE
    fecha_inicial_1 timestamp DEFAULT '2016-10-18 08:00:00';
    fecha_final_1 timestamp DEFAULT '2016-10-18 10:00:00';
    fecha_inicial_2 timestamp DEFAULT '2016-10-18 17:00:00';
    fecha_final_2 timestamp DEFAULT '2016-10-18 19:00:00' ;
    
BEGIN
      PERFORM insert_rows(fecha_inicial_1, fecha_final_1);
      PERFORM insert_rows(fecha_inicial_2, fecha_final_2);
END another_block2 $$;
