DO $$
<<first_block>>
DECLARE 
   t_row vehicle_trajectories_training_modified%rowtype;
   curs1 CURSOR FOR SELECT * FROM vehicle_trajectories_training_modified FOR UPDATE;
   campo link_object;
   link_ids smallint[] DEFAULT '{}';
   routes varchar(100)[];
   contador int DEFAULT '0';
BEGIN
   OPEN curs1;
   LOOP
        FETCH curs1 INTO t_row;
        RAISE NOTICE 'fila : %', t_row.travel_seq;
        EXIT WHEN t_row IS NULL;
        EXECUTE 'SELECT array_agg(link_seq::varchar(100)) FROM vehicle_routes_modified' INTO routes;
        FOREACH campo IN ARRAY t_row.travel_seq
        LOOP
         link_ids := link_ids || campo.id;
        END LOOP;
         RAISE NOTICE 'resultado : %', link_ids;
        contador := contador + 1;
        IF NOT(ARRAY[link_ids::varchar(100)] && routes) THEN
                RAISE EXCEPTION 'Nonexistent route --> %', link_ids;
             END IF;
     link_ids := '{}';
   END LOOP;
   RAISE NOTICE 'TODO CORRECTO';
   CLOSE curs1;
END first_block $$;


