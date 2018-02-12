# DUDAS
* A la hora de realizar la comprobación de las rutas en la columna *travel_seq* de la tabla **vehicle_trajectories_training_modified**, hay rutas que conectan la intersección C con la barrera de peaje 1 pero no se corresponde con la ruta establecida en la tabla **vehicle_routes_modified**. Es decir, hay rutas en las que pasan de un enlace a otro sin tener en cuenta enlaces intermedios. Un ejemplo para ver esto: 

```sql
SELECT *
FROM vehicle_trajectories_training_modified 
WHERE intersection_id = 'C' AND tollgate_id = 1 AND travel_seq[6].id = '121';
```
# SUGERENCIAS
* Se ha dejado al final el volumen de tráfico *out* y el *in* como columnas separadas ya que interesa saber el mismo por separado.

# COSAS QUE ARREGLAR

# OBSERVACIONES
* En los días de entrenamiento proporcionados por la competición hay precipitaciones escasas, por lo que los picos que encontramos en los datos de tiempo promedio de viaje no podemos relacionarlos con las precipitaciones.
