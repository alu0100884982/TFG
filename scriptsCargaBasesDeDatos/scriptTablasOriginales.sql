--Crea la tabla 3: Road link properties
create table road_links(link_id char(3) PRIMARY KEY,
length float,
width float,
lanes int,
in_top varchar(7),
out_top varchar(7),
lane_width float);

COPY road_links  from '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/links_table3.csv' WITH CSV HEADER;

create table vehicle_routes(intersection_id char(1),
tollgate_id char(1),
link_seq varchar(47),
PRIMARY KEY (intersection_id, tollgate_id));

COPY vehicle_routes  from '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/routes_table4.csv' WITH CSV HEADER;

