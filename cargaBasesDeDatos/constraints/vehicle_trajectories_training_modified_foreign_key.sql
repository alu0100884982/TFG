DO $$
<<first_block>>
DECLARE 
   t_row vehicle_trajectories_training_modified%rowtype;
   curs1 CURSOR FOR SELECT * FROM vehicle_trajectories_training_modified FOR UPDATE;
   campo link_object;
   link_ids smallint[];
BEGIN
   OPEN curs1;
   EXECUTE 'SELECT array_agg(link_id) FROM road_links_modified' INTO link_ids;
   LOOP
        FETCH curs1 INTO t_row;
        EXIT WHEN t_row IS NULL;
        FOREACH campo IN ARRAY t_row.travel_seq
        LOOP
             IF NOT(ARRAY[campo.id] && link_ids) THEN
                RAISE EXCEPTION 'Nonexistent link id --> %', campo.id;
             END IF;
        END LOOP;
   END LOOP;
   RAISE NOTICE 'Identificadores de los enlaces válidos en la tabla vehicle_trajectories_training_modified';
   CLOSE curs1;
END first_block $$;
