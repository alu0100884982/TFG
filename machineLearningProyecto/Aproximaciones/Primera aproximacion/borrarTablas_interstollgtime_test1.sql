DO $$
<<block>>
DECLARE
  routes ruta ARRAY;
  tiempos time ARRAY;
  route ruta;
  tiempo time;
BEGIN 

routes := ARRAY(SELECT DISTINCT(intersection_id, tollgate_id)
FROM tabla_resultado_average_travel_time
ORDER BY (intersection_id, tollgate_id));

tiempos := ARRAY(SELECT DISTINCT(time_window[1].time)
FROM tabla_resultado_average_travel_time
ORDER BY time_window[1].time);

FOREACH route IN ARRAY routes LOOP
     FOREACH tiempo IN ARRAY tiempos LOOP
        EXECUTE('DROP TABLE  ' || route.intersection ||'_' ||route.tollgate || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo));
     END LOOP;
END LOOP;
END block $$;

