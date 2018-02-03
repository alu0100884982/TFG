
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

CREATE TABLE vehicle_routes_modified(intersection_id char(1),
tollgate_id smallint,
link_seq varchar(49),
PRIMARY KEY (intersection_id, tollgate_id));

COPY vehicle_routes_modified FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/routes_table4.csv' WITH CSV HEADER;
UPDATE vehicle_routes_modified SET link_seq = STRING_TO_ARRAY(link_seq, ',');
ALTER TABLE vehicle_routes_modified ALTER link_seq type smallint[] using link_seq::smallint[];

CREATE TABLE vehicle_trajectories_training_modified(intersection_id char(1),
tollgate_id smallint,
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
   contador int := 0;
BEGIN
   OPEN curs1;
   LOOP
        contador := contador + 1;
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


