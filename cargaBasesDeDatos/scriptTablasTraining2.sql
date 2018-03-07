CREATE TABLE travel_time_intersection_to_tollgate_training2(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN ('A', 'B', 'C')),
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window varchar(50),
avg_travel_time float);

COPY travel_time_intersection_to_tollgate_training2 FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/dataSet_phase2/20min_avg_travel_time_training2.csv' WITH CSV HEADER;

DO $$
<<second_block>>
DECLARE
   t_row travel_time_intersection_to_tollgate_training2%rowtype;
   curs2 CURSOR FOR SELECT * FROM travel_time_intersection_to_tollgate_training2 FOR UPDATE;
   interval_timestamps timestamp ARRAY[2];
   
BEGIN
   OPEN curs2;
   LOOP
        FETCH curs2 INTO t_row;
        t_row.time_window = regexp_replace( t_row.time_window, '\)|\[', '', 'g');
        interval_timestamps := STRING_TO_ARRAY(t_row.time_window, ',');
        EXIT WHEN t_row IS NULL;
        UPDATE travel_time_intersection_to_tollgate_training2 SET time_window = interval_timestamps
                WHERE CURRENT OF curs2;
   END LOOP;
   CLOSE curs2;
END second_block $$;


ALTER TABLE travel_time_intersection_to_tollgate_training2 ALTER time_window type timestamp ARRAY[2] using time_window::timestamp ARRAY[2];
