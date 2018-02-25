# DUDAS
* A la hora de realizar la comprobación de las rutas en la columna *travel_seq* de la tabla **vehicle_trajectories_training_modified**, hay rutas que conectan la intersección C con la barrera de peaje 1 pero no se corresponde con la ruta establecida en la tabla **vehicle_routes_modified**. Es decir, hay rutas en las que pasan de un enlace a otro sin tener en cuenta enlaces intermedios. Un ejemplo para ver esto: 

```sql
SELECT *
FROM vehicle_trajectories_training_modified 
WHERE intersection_id = 'C' AND tollgate_id = 1 AND travel_seq[6].id = '121';
```
irando si es entre semana o fin de semana) o no y, si fuera recomendable tenerla en cuenta, cómo la uso como variable a la hora de introducirla en el modelo.


* En la tabla resultado de predicción de tiempo promedio de viaje, a la hora de tener en cuenta el tiempo medio de viaje 20 min antes, 40 min antes, etcétera, en los intervalos de tiempo a predecir,  resulta que por ejemplo el intervalo de tiempo  {"2016-10-18 08:20:00","2016-10-18 08:40:00"} necesita, para el atributo 20 min antes, el tiempo promedio de viaje del intervalo anterior ({"2016-10-18 08:00:00","2016-10-18 08:20:00"}), pero este intervalo también está dentro de los intervalos que tengo que predecir y, por tanto, no tengo su tiempo promedio de viaje. ¿Qué debería hacer en este caso? ¿Hacer la media de todos los tiempos promedios de viaje en el intervalo de tiempo (08:00:00 - 08:20:00) en los dias anteriores en la ruta que estemos considerando?
# SUGERENCIAS
* Se ha dejado al final el volumen de tráfico *out* y el *in* como columnas separadas ya que interesa saber el mismo por separado.

# COSAS QUE ARREGLAR


# OBSERVACIONES
* En los días de entrenamiento proporcionados por la competición hay precipitaciones escasas, por lo que los picos que encontramos en los datos de tiempo promedio de viaje no podemos relacionarlos con las precipitaciones.
* El tipo de vehículo no lo proporciona la competición, por lo que descartamos este atributo.
