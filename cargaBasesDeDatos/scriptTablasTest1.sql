CREATE TABLE travel_time_intersection_to_tollgate_test1(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN ('A', 'B', 'C')),
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window varchar(50),
avg_travel_time float);

COPY travel_time_intersection_to_tollgate_test1 FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/testing_phase1/test1_20min_avg_travel_time.csv' WITH CSV HEADER;

DO $$
<<second_block>>
DECLARE
   t_row travel_time_intersection_to_tollgate_test1%rowtype;
   curs2 CURSOR FOR SELECT * FROM travel_time_intersection_to_tollgate_test1 FOR UPDATE;
   interval_timestamps timestamp ARRAY[2];
   
BEGIN
   OPEN curs2;
   LOOP
        FETCH curs2 INTO t_row;
        t_row.time_window = regexp_replace( t_row.time_window, '\)|\[', '', 'g');
        interval_timestamps := STRING_TO_ARRAY(t_row.time_window, ',');
        EXIT WHEN t_row IS NULL;
        UPDATE travel_time_intersection_to_tollgate_test1 SET time_window = interval_timestamps
                WHERE CURRENT OF curs2;
   END LOOP;
   CLOSE curs2;
END second_block $$;


ALTER TABLE travel_time_intersection_to_tollgate_test1 ALTER time_window type timestamp ARRAY[2] using time_window::timestamp ARRAY[2];

/**************************************************************************************************************************************************/
CREATE TABLE tabla_resultado_average_travel_time(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN ('A', 'B', 'C')),
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window timestamp ARRAY[2],
twenty_min_previous float,
forty_min_previous float,
sixty_min_previous float,
eighty_min_previous float,
onehundred_min_previous float,
onehundredandtwenty_min_previous float,
avg_travel_time float);

CREATE TYPE route AS (
    intersection_id  text,
    tollgate_id      integer
);

DO $$
<<time_travel_block>>
DECLARE
    rutas  route[6];
    fecha_inicial_1 timestamp DEFAULT '2016-10-18 08:00:00';
    fecha_final_1 timestamp DEFAULT '2016-10-18 10:00:00';
    fecha_inicial_2 timestamp DEFAULT '2016-10-18 17:00:00';
    fecha_final_2 timestamp DEFAULT '2016-10-24 19:00:00' ;
    ruta route;
    aux1 timestamp;
    contador integer;
    
BEGIN
      rutas[1] = ('A',2);
      rutas[2] = ('A',3);
      rutas[3] = ('B',1);
      rutas[4] = ('B',3);
      rutas[5] = ('C',1);
      rutas[6] = ('C',3);
      
      FOREACH ruta IN ARRAY rutas
      LOOP
         fecha_inicial_1 := '2016-10-18 08:00:00';
         fecha_final_1 := '2016-10-18 10:00:00';
         FOR contador in 1..7 LOOP
            aux1 := fecha_inicial_1;
         WHILE aux1 != fecha_final_1 LOOP
           INSERT INTO tabla_resultado_average_travel_time VALUES(ruta.intersection_id,ruta.tollgate_id, ARRAY[aux1, aux1 + '20 minute'],NULL, NULL,NULL,NULL,NULL,NULL,NULL);
           aux1 :=  aux1 + '20 minute';
         END LOOP;
         fecha_inicial_1 := fecha_inicial_1 + '1 day';
          fecha_final_1 := fecha_final_1 + '1 day';
        END LOOP;
      END LOOP;
      
        FOREACH ruta IN ARRAY rutas
      LOOP
         fecha_inicial_2 := '2016-10-18 17:00:00';
         fecha_final_2 := '2016-10-18 19:00:00';
         FOR contador in 1..7 LOOP
            aux1 := fecha_inicial_2;
         WHILE aux1 != fecha_final_2 LOOP
           INSERT INTO tabla_resultado_average_travel_time VALUES(ruta.intersection_id,ruta.tollgate_id, ARRAY[aux1, aux1 + '20 minute'],NULL, NULL,NULL,NULL,NULL,NULL,NULL);
           aux1 :=  aux1 + '20 minute';
         END LOOP;
         fecha_inicial_2 := fecha_inicial_2 + '1 day';
          fecha_final_2 := fecha_final_2 + '1 day';
        END LOOP;
      END LOOP;
END time_travel_block $$;


/****************************************************************************************************************************************/
CREATE TABLE tabla_resultado_traffic_volume (tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window timestamp ARRAY[2],
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
/********************************************************************************************/
CREATE TABLE traffic_volume_tollgates_test1 (tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window varchar(50),
direction smallint CONSTRAINT has_direction_value CHECK (direction IN (0,1)),
volume int,
proportion_hasetc_vehicles int);

COPY traffic_volume_tollgates_test1 FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/testing_phase1/test1_20min_avg_volume.csv' WITH CSV HEADER;

DO $$
<<third_block>>
DECLARE
   t_row traffic_volume_tollgates_test1%rowtype;
   curs3 CURSOR FOR SELECT * FROM traffic_volume_tollgates_test1 FOR UPDATE;
   interval_timestamps timestamp ARRAY[2];
   
BEGIN
   OPEN curs3;
   LOOP
        FETCH curs3 INTO t_row;
        t_row.time_window = regexp_replace( t_row.time_window, '\)|\[', '', 'g');
        interval_timestamps := STRING_TO_ARRAY(t_row.time_window, ',');
        EXIT WHEN t_row IS NULL;
        UPDATE traffic_volume_tollgates_test1 SET time_window = interval_timestamps
                WHERE CURRENT OF curs3;
   END LOOP;
   CLOSE curs3;
END third_block $$;

ALTER TABLE  traffic_volume_tollgates_test1 ALTER time_window type timestamp ARRAY[2] using time_window::timestamp ARRAY[2];


/****Cálculo del tiempo promedio de viaje dos horas antes en la tabla tabla_resultado_average_travel_time**/

UPDATE tabla_resultado_average_travel_time t
SET two_hours_previous = (SELECT AVG(avg_travel_time) FROM travel_time_intersection_to_tollgate_test1 s
WHERE s.intersection_id = t.intersection_id AND s.tollgate_id = t.tollgate_id AND            
(s.time_window[1] BETWEEN t.time_window[1] - INTERVAL '2 hours' AND t.time_window[1] - INTERVAL'20 min') AND  
(s.time_window[2] BETWEEN t.time_window[2] - INTERVAL '2 hours' AND t.time_window[2]- INTERVAL '20 min'))
;
/*************************************************************/

CREATE TABLE weather_data_test1 (date_ date, 
hour int CONSTRAINT correct_hour CHECK (hour BETWEEN 0 AND 23),
pressure float,
sea_pressure float,
wind_direction float CONSTRAINT degrees CHECK (wind_direction BETWEEN 0 AND 360),
wind_speed float,
temperature float,
rel_humidity float CONSTRAINT correct_humidity CHECK (rel_humidity BETWEEN 0 AND 100),
precipitation float);

COPY weather_data_test1 FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/testing_phase1/weather (table 7)_test1.csv' WITH CSV HEADER;
