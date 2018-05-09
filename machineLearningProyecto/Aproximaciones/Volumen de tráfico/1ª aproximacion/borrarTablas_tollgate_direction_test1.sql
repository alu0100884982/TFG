DO $$
<<block>>
DECLARE
  pares par ARRAY;
  tiempos time ARRAY;
  pair  par;
  tiempo time;
BEGIN 

pares := ARRAY(SELECT DISTINCT(tollgate_id, direction)
FROM tabla_resultado_traffic_volume
ORDER BY (tollgate_id, direction));

tiempos := ARRAY(SELECT DISTINCT(time_window[1].time)
FROM tabla_resultado_traffic_volume
ORDER BY time_window[1].time);

FOREACH pair  IN ARRAY pares LOOP
     FOREACH tiempo IN ARRAY tiempos LOOP
        EXECUTE('DROP TABLE volume_' || pair.tollgate ||'_' ||pair.direction || '_' ||EXTRACT(HOUR FROM tiempo) || '_' ||EXTRACT(MINUTE FROM tiempo));
     END LOOP;
END LOOP;
END block $$;

