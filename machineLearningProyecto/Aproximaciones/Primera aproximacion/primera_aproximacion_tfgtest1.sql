CREATE OR REPLACE VIEW weather_byday AS SELECT  date_, AVG(pressure) as pressure, AVG(sea_pressure) as sea_pressure, AVG(wind_direction) as wind_direction, AVG(wind_speed) as wind_speed, AVG(temperature) as temperature, AVG(rel_humidity) as rel_humidity, AVG(precipitation) as precipitation
FROM weather_data_modified GROUP by date_
ORDER BY date_;

DO $$
<<block>>
DECLARE

    
BEGIN 
   EXECUTE('CREATE OR REPLACE VIEW vista_travel_time_contiempometeorologico AS SELECT *
FROM weather_byday w JOIN  tabla_resultado_average_travel_time t
ON w.date_ = t.time_window[1].date;');
      
END block $$;
