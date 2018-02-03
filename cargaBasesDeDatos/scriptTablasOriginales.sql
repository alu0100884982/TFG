--Crea la tabla 3: Road link properties
CREATE TABLE road_links(link_id char(3) PRIMARY KEY,
length float,
width float,
lanes int,
in_top varchar(7),
out_top varchar(7),
lane_width float);

COPY road_links  FROM  '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/links_table3.csv' WITH CSV HEADER;

CREATE TABLE vehicle_routes(intersection_id char(1),
tollgate_id char(1),
link_seq varchar(47),
PRIMARY KEY (intersection_id, tollgate_id));

COPY vehicle_routes  FROM  '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/routes_table4.csv' WITH CSV HEADER;

CREATE TABLE vehicle_trajectories_training(intersection_id char(1),
tollgate_id char(1),
vehicle_id varchar(30),
starting_time timestamp,
travel_seq varchar(400),
travel_time float);

COPY vehicle_trajectories_training FROM  '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/trajectories_table 5_training.csv' WITH CSV HEADER;

CREATE TABLE traffic_volume_tollgates_training(time timestamp,
tollgate_id char(1),
direction char(1),
vehicle_model int CONSTRAINT vehicle_model_value CHECK (vehicle_model BETWEEN 0 AND 7),
has_etc char(1) CONSTRAINT has_etc_value CHECK (has_etc IN ('0','1')),
vehicle_type char(1));

COPY traffic_volume_tollgates_training FROM  '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/volume_table 6_training.csv' WITH CSV HEADER;

CREATE TABLE weather_data (date_ date, 
hour int,
pressure float,
sea_pressure float,
wind_direction float,
wind_speed float,
temperature float,
rel_humidity float,
precipitation float);

COPY weather_data FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/weather (table 7)_training.csv' WITH CSV HEADER;

CREATE TABLE travel_time_intersection_to_tollgate (intersection_id char(1),
tollgate_id char(1),
time_window varchar(43),
avg_travel_time float);

COPY travel_time_intersection_to_tollgate FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/trajectories_table5_training_20min_avg_travel_time.csv' WITH CSV HEADER;

CREATE TABLE traffic_volume_tollgates (tollgate_id char(1),
time_window varchar(45),
direction char(1),
volume int);

COPY traffic_volume_tollgates FROM '/home/javisunami/Escritorio/TFG/datasetsOriginales/training/volume_table 6_training_20min_avg_volume.csv' WITH CSV HEADER;
