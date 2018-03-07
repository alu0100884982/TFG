SELECT intersection_id, tollgate_id,vehicle_id, ROUND(ABS((SELECT SUM(s.duration) FROM UNNEST(travel_seq) s) - travel_time)::numeric, 2) 
FROM vehicle_trajectories_training_modified ;
