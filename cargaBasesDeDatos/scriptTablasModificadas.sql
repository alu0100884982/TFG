
CREATE TABLE road_links_modified(link_id smallint PRIMARY KEY,
length float,
width float,
lanes int,
in_top varchar(9),
out_top varchar(9),
lane_width float);


COPY road_links_modified  FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/links_table3.csv' WITH CSV HEADER;
UPDATE road_links_modified SET in_top = STRING_TO_ARRAY(in_top, ',');
UPDATE road_links_modified SET out_top = STRING_TO_ARRAY(out_top, ',');
ALTER TABLE road_links_modified ALTER in_top type smallint[] using in_top::smallint[];
ALTER TABLE road_links_modified ALTER out_top type smallint[] using out_top::smallint[];

CREATE TABLE vehicle_routes_modified(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN ('A', 'B', 'C')),
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id IN (1,2,3)),
link_seq varchar(49),
PRIMARY KEY (intersection_id, tollgate_id));

COPY vehicle_routes_modified FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/routes_table4.csv' WITH CSV HEADER;
UPDATE vehicle_routes_modified SET link_seq = STRING_TO_ARRAY(link_seq, ',');
ALTER TABLE vehicle_routes_modified ALTER link_seq type smallint[] using link_seq::smallint[];

CREATE TABLE vehicle_trajectories_training_modified(intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN ('A', 'B', 'C')),
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id IN (1,2,3)),
vehicle_id int,
starting_time timestamp,
travel_seq varchar(402),
travel_time float);

COPY vehicle_trajectories_training_modified from '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/trajectories_table 5_training.csv' WITH CSV HEADER;
UPDATE vehicle_trajectories_training_modified SET travel_seq = STRING_TO_ARRAY(travel_seq, ';');

ALTER TABLE vehicle_trajectories_training_modified ALTER travel_seq type varchar(100) ARRAY using travel_seq::varchar(100) ARRAY;

CREATE TYPE link_object AS (
 id smallint,
 entrance_time timestamp,
 duration float
);


DO $$
<<first_block>>
DECLARE 
   t_row vehicle_trajectories_training_modified%rowtype;
   curs1 CURSOR FOR SELECT * FROM vehicle_trajectories_training_modified FOR UPDATE;
   link link_object;
   conjunto_links link_object[] DEFAULT '{}';
   campo varchar(100);
   array_componentes_campo varchar(40) ARRAY;
BEGIN
   OPEN curs1;
   LOOP
        FETCH curs1 INTO t_row;
        EXIT WHEN t_row IS NULL;
        FOREACH campo IN ARRAY t_row.travel_seq
        LOOP
              array_componentes_campo := STRING_TO_ARRAY(campo, '#');
              link.id :=  array_componentes_campo[1];
              link.entrance_time :=  array_componentes_campo[2];
              link.duration :=  array_componentes_campo[3];
              conjunto_links := conjunto_links || link;
        END LOOP;
        UPDATE vehicle_trajectories_training_modified SET travel_seq = conjunto_links
                WHERE CURRENT OF curs1;
        conjunto_links := '{}';
        link = null;
   END LOOP;
   CLOSE curs1;
END first_block $$;

ALTER TABLE vehicle_trajectories_training_modified ALTER travel_seq type link_object[] using travel_seq::link_object[];


CREATE TABLE traffic_volume_tollgates_training_modified(time timestamp,
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
direction smallint CONSTRAINT has_direction_value CHECK (direction IN (0,1)),
vehicle_model int CONSTRAINT vehicle_model_value CHECK (vehicle_model BETWEEN 0 AND 7),
has_etc boolean CONSTRAINT has_etc_value CHECK (has_etc IN ('0','1')),
vehicle_type char(1));

COPY traffic_volume_tollgates_training_modified FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/volume_table 6_training.csv' WITH CSV HEADER;
ALTER TABLE traffic_volume_tollgates_training_modified DROP COLUMN vehicle_model, DROP COLUMN vehicle_type;

CREATE TABLE weather_data_modified (date_ date, 
hour int,
pressure float,
sea_pressure float,
wind_direction float,
wind_speed float,
temperature float,
rel_humidity float,
precipitation float);

COPY weather_data_modified FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/weather (table 7)_training.csv' WITH CSV HEADER;

CREATE TABLE travel_time_intersection_to_tollgate_modified (intersection_id char(1) CONSTRAINT has_intersection_id_value CHECK (intersection_id IN ('A', 'B', 'C')),
tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window varchar(50),
avg_travel_time float);

COPY travel_time_intersection_to_tollgate_modified FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/trajectories_table5_training_20min_avg_travel_time.csv' WITH CSV HEADER;

DO $$
<<second_block>>
DECLARE
   t_row travel_time_intersection_to_tollgate_modified%rowtype;
   curs2 CURSOR FOR SELECT * FROM travel_time_intersection_to_tollgate_modified FOR UPDATE;
   interval_timestamps timestamp ARRAY[2];
   
BEGIN
   OPEN curs2;
   LOOP
        FETCH curs2 INTO t_row;
        t_row.time_window = regexp_replace( t_row.time_window, '\)|\[', '', 'g');
        interval_timestamps := STRING_TO_ARRAY(t_row.time_window, ',');
        EXIT WHEN t_row IS NULL;
        UPDATE travel_time_intersection_to_tollgate_modified SET time_window = interval_timestamps
                WHERE CURRENT OF curs2;
   END LOOP;
   CLOSE curs2;
END second_block $$;


ALTER TABLE travel_time_intersection_to_tollgate_modified ALTER time_window type timestamp ARRAY[2] using time_window::timestamp ARRAY[2];

CREATE TABLE traffic_volume_tollgates_modified (tollgate_id smallint CONSTRAINT has_tollgate_id_value CHECK (tollgate_id  IN (1,2,3)),
time_window varchar(50),
direction smallint CONSTRAINT has_direction_value CHECK (direction IN (0,1)),
volume int);

COPY traffic_volume_tollgates_modified FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/volume_table 6_training_20min_avg_volume.csv' WITH CSV HEADER;

DO $$
<<third_block>>
DECLARE
   t_row traffic_volume_tollgates_modified%rowtype;
   curs3 CURSOR FOR SELECT * FROM traffic_volume_tollgates_modified FOR UPDATE;
   interval_timestamps timestamp ARRAY[2];
   
BEGIN
   OPEN curs3;
   LOOP
        FETCH curs3 INTO t_row;
        t_row.time_window = regexp_replace( t_row.time_window, '\)|\[', '', 'g');
        interval_timestamps := STRING_TO_ARRAY(t_row.time_window, ',');
        EXIT WHEN t_row IS NULL;
        UPDATE traffic_volume_tollgates_modified SET time_window = interval_timestamps
                WHERE CURRENT OF curs3;
   END LOOP;
   CLOSE curs3;
END third_block $$;

ALTER TABLE  traffic_volume_tollgates_modified ALTER time_window type timestamp ARRAY[2] using time_window::timestamp ARRAY[2];



